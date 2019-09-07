import sys
from pathlib import Path
from typing import Optional, Union

import tqdm
import pyaml
import numpy as np
import pandas as pd
import torch.utils.data
import torchgraphs as tg

from . import features

MAX_CUTOFF_DISTANCE = 12


class ProteinQualityDataset(torch.utils.data.Dataset):
    def __init__(self, samples: pd.DataFrame, transforms=()):
        """Load graph samples in `*.pt` format from a folder
        :param samples: a pandas dataframe with columns {'target', 'model', 'path'}
        :param transforms: transformations to apply to every sample
        """
        self.samples = samples
        self.transforms = transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        path = self.samples.path.iloc[item]
        sample = torch.load(path)  # type: (str, str, tg.Graph, tg.Graph)

        for t in self.transforms:
            sample = t(*sample)

        return sample


class SelectNodeFeatures(object):
    def __init__(self, residues=True, partial_entropy=True, self_info=True, dssp_features=True):
        self.column_mask = torch.full((features.Input.Node.LENGTH,), fill_value=False, dtype=torch.bool)
        if residues:
            self.column_mask[features.Input.Node.RESIDUES] = True
        if partial_entropy:
            self.column_mask[features.Input.Node.PARTIAL_ENTROPY] = True
        if self_info:
            self.column_mask[features.Input.Node.SELF_INFO] = True
        if dssp_features:
            self.column_mask[features.Input.Node.DSSP_FEATURES] = True
        if self.num_features == 0:
            raise ValueError('No node features selected.')

    @property
    def num_features(self):
        return self.column_mask.int().sum().item()

    def __call__(self, protein: str, provider: str, graph_in: tg.Graph, graph_target: tg.Graph):
        graph_in = graph_in.evolve(node_features=graph_in.node_features[:, self.column_mask])
        return protein, provider, graph_in, graph_target


class RemoveEdges(object):
    def __init__(self, cutoff: float):
        """
        :param cutoff: the maximum distance at which non-adjacent residues should be connected,
                       passing 0 will result in simple linear graph structure
        """
        if cutoff >= MAX_CUTOFF_DISTANCE:
            from warnings import warn
            warn(f'The chosen cutoff {cutoff} is larger than the maximum distance '
                 f'saved in the dataset {MAX_CUTOFF_DISTANCE}, this transformation will not remove any edge')
        self.cutoff = cutoff

    def __call__(self, protein: str, provider: str, graph_in: tg.Graph, graph_target: tg.Graph):
        # Remove edges between non-adjacent residues with distance greater than cutoff
        if self.cutoff < MAX_CUTOFF_DISTANCE:
            distances = graph_in.edge_features[:, features.Input.Edge.SPATIAL_DISTANCE]
            is_peptide_bond = graph_in.edge_features[:, features.Input.Edge.IS_PEPTIDE_BOND]
            to_keep = (is_peptide_bond > 0) | (distances < self.cutoff)

            graph_in = graph_in.evolve(
                senders=graph_in.senders[to_keep],
                receivers=graph_in.receivers[to_keep],
                edge_features=graph_in.edge_features[to_keep],
            )

        return protein, provider, graph_in, graph_target


class RbfDistEdges(object):
    def __init__(self, sigma: float):
        if sigma <= 0:
            raise ValueError(f'RBF variance must be strictly positive, got {sigma}')
        self.sigma = sigma

    def __call__(self, protein: str, provider: str, graph_in: tg.Graph, graph_target: tg.Graph):
        # Distances are encoded using a 0-centered RBF with variance sigma
        distances = graph_in.edge_features[:, features.Input.Edge.SPATIAL_DISTANCE]
        distances_rbf = torch.exp(- distances ** 2 / (2 * self.sigma))

        graph_in = graph_in.evolve(
            edge_features=torch.cat((
                distances_rbf[:, None],
                graph_in.edge_features[:, features.Input.Edge.SEPARATION_OH]
            ), dim=1),
        )

        return protein, provider, graph_in, graph_target


class SeparationEncoding(object):
    def __init__(self, use_separation: bool):
        self.use_separation = use_separation

    def __call__(self, protein: str, provider: str, graph_in: tg.Graph, graph_target: tg.Graph):
        if not self.use_separation:
            feature_columns = [features.Input.Edge.SPATIAL_DISTANCE, features.Input.Edge.IS_PEPTIDE_BOND]
            graph_in = graph_in.evolve(
                edge_features=graph_in.edge_features[:, feature_columns]
            )

        return protein, provider, graph_in, graph_target


class PositionalEncoding(object):
    def __init__(self, encoding_size, max_sequence_length, base, device='cpu'):
        self.encoding_size = encoding_size
        self.max_sequence_length = max_sequence_length
        self.base = base

        if self.encoding_size != 0 and base is not None:
            self.encoding = self.make_encoding(encoding_size, max_sequence_length, base, device)

    def __call__(self, protein: str, provider: str, graph_in: tg.Graph, graph_target: tg.Graph):
        if self.encoding_size == 0:
            return protein, provider, graph_in, graph_target

        graph_in = graph_in.evolve(node_features=torch.cat((
            graph_in.node_features,
            self.encoding[:graph_in.num_nodes, :]
        ), dim=1))
        return protein, provider, graph_in, graph_target

    @staticmethod
    def make_encoding(encoding_size: int, len_sequence: int, base: float = 10000,
                      device: Optional[Union[str, torch.device]] = 'cpu'):
        """Prepare a positional encoding of shape (len_sequence, dim_encoding)

        Args:
            encoding_size:
            len_sequence:
            base:
            device:

        Returns:

        """
        # sequence_pos, encoding_idx have both shape (len_sequence, dim_encoding)
        sequence_pos, encoding_idx = torch.meshgrid(
            torch.arange(len_sequence, device=device, dtype=torch.float),
            torch.arange(encoding_size, device=device, dtype=torch.float)
        )
        enc = torch.empty(len_sequence, encoding_size, device=device)
        enc[:, 0::2] = torch.sin(sequence_pos[:, 0::2] / (base ** (2 * encoding_idx[:, 0::2] / encoding_size)))
        enc[:, 1::2] = torch.cos(sequence_pos[:, 1::2] / (base ** (2 * encoding_idx[:, 1::2] / encoding_size)))
        return enc


def process_file(filepath, destpath):
    import tables

    filepath = Path(filepath).expanduser().resolve()
    h5_file = tables.open_file(filepath)

    destpath = Path(destpath).expanduser().resolve()
    destpath.mkdir(parents=True, exist_ok=False)

    scores = {
        'local_lddt': [],
        'local_sscore': [],
        'global_lddt': [],
        'global_gdtts': [],
        'global_gdtts_ha': [],
        'global_rmsd': [],
        'global_maxsub': [],
    }
    sequence_lengths = []
    protein_count = 0
    model_count = 0

    # First pass: enumerate proteins, count models, prepare frequency stats
    proteins = []
    for protein in h5_file.list_nodes('/'):
        # I'm not sure why the file contains these nodes, but they are not proteins for sure
        protein_name = protein._v_name
        if protein_name.startswith(('casp', 'cameo')):
            continue

        num_models = len(protein.names)
        if num_models <= 5:
            print(f'Skipping target {protein_name} with only {num_models} model(s).', file=sys.stderr)
            continue

        if np.any(np.isnan(protein.lddt_global)):
            print(f'Skipping target {protein_name} with {np.count_nonzero(np.isnan(protein.lddt_global))}/'
                  f'{len(protein.lddt_global)} missing global LDDT scores.', file=sys.stderr)
            continue

        if np.any(np.isnan(protein.gdt_ts)):
            print(f'Skipping target {protein_name} with {np.count_nonzero(np.isnan(protein.lddt_global))}/'
                  f'{len(protein.lddt_global)} missing global GDT_TS scores.', file=sys.stderr)
            continue

        scores['local_lddt'].append(np.array(protein.lddt).ravel())
        scores['local_sscore'].append(np.array(protein.s_scores).ravel())
        scores['global_lddt'].append(np.array(protein.lddt_global))
        scores['global_gdtts'].append(np.array(protein.gdt_ts))
        scores['global_gdtts_ha'].append(np.array(protein.gdt_ts_ha))
        scores['global_rmsd'].append(np.array(protein.rmsd))
        scores['global_maxsub'].append(np.array(protein.max_sub))

        protein_count += 1
        sequence_lengths.append(protein.seq.shape[1])
        model_count += num_models
        proteins.append(protein)

        Path.mkdir(destpath / protein_name, exist_ok=False)
        with open(destpath / protein_name / 'protein_stats.yaml', 'w') as f:
            pyaml.dump({
                'protein': protein_name,
                'num_models': len(protein.names),
                'length': protein.seq.shape[1],
            }, dst=f, sort_dicts=False)

    # Bucketize the scores into 20 equally-spaced bins in the range [0, 1] and compute frequency counts for each bin.
    # Compute weights for the local scores proportionally to the inverse of the frequency of
    # that score in the dataset, set minimum frequency to 1 to avoid +inf weights
    weights = {}
    bins = np.linspace(0, 1, num=20 + 1)
    for name in scores.keys():
        scores[name] = pd.Series(np.concatenate(scores[name]), name=name).dropna()
        frequencies = pd.cut(scores[name], bins=bins, include_lowest=True).value_counts().sort_index()
        frequencies.to_pickle(destpath / f'{name}_frequencies.pkl')
        weights[name] = frequencies.max() / frequencies.clip(lower=1)

    dataset_stats = {
        'dataset': Path(filepath).name,
        'num_proteins': protein_count,
        'num_models': model_count,
        'avg_length': sum(sequence_lengths) / protein_count,
        'max_length': max(sequence_lengths),
        **{score_name: scores_series.describe().to_dict() for score_name, scores_series in scores.items()}
    }
    pyaml.print(dataset_stats, sort_dicts=False)
    with open(destpath / 'dataset_stats.yaml', 'w') as f:
        pyaml.dump(dataset_stats, dst=f, sort_dicts=False)

    # Second pass: save every model as a pytorch object
    models_bar = tqdm.tqdm(total=model_count, desc='Models  ', unit=' models', leave=False)
    samples_df = []
    for protein in proteins:
        for model_idx in range(len(protein.names)):
            sample = protein_model_to_graph(protein, model_idx, weights)
            torch.save(sample, destpath / sample[0] / f'{sample[1]}.pt')
            samples_df.append({
                'target': sample[0],
                'model': sample[1],
                'path': f'{sample[0]}/{sample[1]}.pt'
            })
            models_bar.update()
    pd.DataFrame(samples_df).to_csv(destpath / 'samples.csv', header=True, index=False)
    models_bar.close()
    h5_file.close()


def protein_model_to_graph(protein, model_idx, weights):
    # Name of this protein
    protein_name = protein._v_name

    # Protein sequence and multi-sequence-alignment features, common to all models
    residues = protein.seq[0]                # One-hot encoding of the residues present in this model
    partial_entropy = protein.part_entr[0]   # Partial entropy of the residues
    self_information = protein.self_info[0]  # Self information of the residues
    sequence_length = len(residues)

    # Secondary structure is determined using some method by some research group.
    # All models use the original sequence as source, but can fail to determine a structure for some residues.
    # The DSSP features are extracted for the residues whose 3D coordinates are determined.
    # Missing DSSP features are filled with 0s in the dataset.
    decoy_name = protein.names[model_idx].decode('utf-8')  # Who built this model of the protein
    coordinates = protein.cb_coordinates[model_idx]        # Coordinates of the β carbon (α if β is not present)
    structure_determined = protein.valid_dssp[model_idx]   # Which residues are determined within this model
    dssp_features = protein.dssp[model_idx]                # DSSP features for the secondary structure

    # For every residue, a score is determined by comparison with the native structure (using LDDT)
    # For some models, it is not possible to assign a score to a residue if:
    # - the experiment to determine the native model has failed to  _observe_ the secondary structure of that residue
    # - the algorithm to determine this model has failed to _determine_ the secondary structure of that residue
    # Frequency weights are looked up in the corresponding weights table (NaN scores get NaN weight).
    local_lddt = protein.lddt[model_idx]

    local_lddt_valid = np.array(protein.valid[model_idx], dtype=np.bool)
    if np.any(local_lddt_valid != np.isfinite(local_lddt)):
        tqdm.tqdm.write(f'Inconsistent validity of local LDDT scores for {protein_name}/{decoy_name}: '
                        f'found {np.count_nonzero(np.isfinite(local_lddt))} not null scores, '
                        f'expected {np.count_nonzero(local_lddt_valid)} valid scores.', file=sys.stderr)
        if decoy_name == 'native':
            # Native models report a score of 1. for all residues, but some of them
            # were not determined, so those should be marked invalid.
            local_lddt[~local_lddt_valid] = np.nan
        else:
            # Why does this happen? David says:
            # "It may be that the C alpha are present, so we can evaluate similarity,
            #  but not all the backbone is, so dssp cannot compute the angles."
            local_lddt_valid = np.isfinite(local_lddt)

    local_lddt_weights = np.full_like(local_lddt, fill_value=np.nan)
    local_lddt_weights[local_lddt_valid] = weights['local_lddt'].loc[local_lddt[local_lddt_valid]].values
    if np.isnan(local_lddt[local_lddt_valid]).any() or np.isnan(local_lddt_weights[local_lddt_valid]).any():
        raise ValueError(f'Found NaN values in local LDDT scores for {protein_name}/{decoy_name}')

    # The global LDDT score is an average of the local LDDT scores:
    # - residues missing from the model get a score of 0
    # - residues missing from the native structure are ignored in the average
    # Frequency weight us looked up in the corresponding weights table.
    global_lddt = protein.lddt_global[model_idx]
    global_lddt_weight = weights['global_lddt'].loc[global_lddt]
    if np.isnan(global_lddt) or np.isnan(global_lddt_weight):
        raise ValueError(f'Found NaN values in global LDDT score for {protein_name}/{decoy_name}')

    # The global GDT_TS score is another way of scoring the quality of a model.
    # Until we get GDT_TS in the dataset we use an average of the local S-score as a proxy for GDT_TS.
    # This is ok because GDT_TS and S-scores have a Pearson correlation of 1.
    # Since the S-score of missing residues is NaN, we ignore those scores when taking the mean.
    # Frequency weight is looked up in the corresponding weights table.
    global_gdtts = protein.gdt_ts[model_idx]
    global_gdtts_weight = weights['global_gdtts'].loc[global_gdtts]
    if np.isnan(global_gdtts) or np.isnan(global_gdtts_weight):
        raise ValueError(f'Found NaN values in global GDT_TS score for {protein_name}/{decoy_name}')

    senders, receivers, edge_features = make_edges(coordinates)

    graph_in = tg.Graph(
        num_nodes=sequence_length,
        node_features=torch.from_numpy(np.concatenate([
            residues,
            partial_entropy,
            self_information,
            dssp_features,
            structure_determined[:, None] * 2 - 1
        ], axis=1)).float(),
        senders=torch.from_numpy(np.concatenate((senders, receivers))),
        receivers=torch.from_numpy(np.concatenate((receivers, senders))),
        edge_features=torch.from_numpy(edge_features).repeat(2, 1).float()
    ).validate()

    graph_target = tg.Graph(
        num_nodes=sequence_length,
        node_features=torch.from_numpy(np.stack([
            local_lddt,
            local_lddt_weights,
        ], axis=1)).float(),
        global_features=torch.tensor([
            global_lddt,
            global_gdtts,
            global_lddt_weight,
            global_gdtts_weight,
        ]).float()
    ).validate()

    assert torch.isfinite(graph_in.node_features).all()
    assert torch.isfinite(graph_in.edge_features).all()

    return protein_name, decoy_name, graph_in, graph_target


def make_edges(coords):
    import pandas as pd
    import scipy.spatial.distance

    # The distances are returned in a condensed upper triangular form without the diagonal,
    # e.g. for 5 coordinates, we would have:
    # distances = [dist(p[0], p[1]), dist(p[0], p[2]), dist(p[0], p[3]), ..., dist(p[3], p[4])]
    # senders   = [0, 0, 0, 0, 1, 1, 1, 2, 2, 3]
    # receivers = [1, 2, 3, 4, 2, 3, 4, 3, 4, 4]
    distances = scipy.spatial.distance.pdist(coords)
    senders, receivers = np.triu_indices(len(coords), k=1)

    # Separation = number of residues between two residues in the sequence.
    # Chemically bonded adjacent residues have 0 separation.
    # Separation is bucketized in 6 categories according to [0, 1, 2, 3, 4, 5:10, 10:]
    separation = receivers - senders - 1
    bins = [0, 1, 2, 3, 4, 5, 10]
    separation_enc = np.digitize(separation, bins=bins, right=False) - 1
    separation_onehot = pd.get_dummies(
        pd.Categorical(separation_enc, categories=np.arange(len(bins)), ordered=True)).values

    # The spatial distance between adjacent residues might be missing (NaN) because the residues don't have
    # 3D coordinates in this particular model. For adjacent residues (separation = 0) we set the edge length
    # to a random value with mean and variance equal to the mean and variance of the distance between
    # adjacent residues in the whole dataset. For residues further apart (separation > 0) we leave NaN.
    to_fill = (separation == 0) & (np.isnan(distances))
    distances[to_fill] = np.random.normal(5.349724573740155, 0.9130922391969375, np.count_nonzero(to_fill))

    edge_features = np.concatenate((
        distances[:, None],
        separation_onehot
    ), axis=1)

    # Distances greater that 12 Angstrom are considered irrelevant and removed unless between adjacent residues.
    with np.errstate(invalid='ignore'):
        to_keep = (distances < MAX_CUTOFF_DISTANCE) | (separation == 0)

    return senders[to_keep], receivers[to_keep], edge_features[to_keep]


def describe(filepath):
    import tables
    import tqdm
    import scipy.spatial
    from proteins.utils import RunningStats

    h5_file = tables.open_file(filepath)

    all_distances = RunningStats()
    all_neighbor_distances = RunningStats()
    for protein in tqdm.tqdm(h5_file.list_nodes('/')):
        for model_idx in tqdm.trange(len(protein.names), desc='Proteins', unit=' proteins'):
            # which residues are actually present in this model
            residue_mask = protein.valid_dssp[model_idx]
            # coordinates of the β carbon (or α if β is not present)
            coordinates = protein.cb_coordinates[model_idx][residue_mask]
            # flattened pairwise distances between residues
            distances = scipy.spatial.distance.pdist(coordinates)
            all_distances.add_from(distances)
            neighbor_distances = np.sqrt(np.square(coordinates[:-1] - coordinates[1:]).sum(axis=1))
            all_neighbor_distances.add_from(neighbor_distances)
    print(all_distances)
    print(all_neighbor_distances)

    h5_file.close()


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    sp_preprocess = subparsers.add_parser('preprocess', help='Process protein file')
    sp_preprocess.add_argument('--filepath', type=str, required=True)
    sp_preprocess.add_argument('--destpath', type=str, required=True)
    sp_preprocess.set_defaults(command=process_file)

    sp_describe = subparsers.add_parser('describe', help='Describe existing datasets')
    sp_describe.add_argument('--filepath', type=str, required=True)
    sp_describe.set_defaults(command=describe)

    args = parser.parse_args()
    args.command(**{k: v for k, v in vars(args).items() if k != 'command'})


if __name__ == '__main__':
    main()
