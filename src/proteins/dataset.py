from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch.utils.data
import torchgraphs as tg

from . import features

MAX_CUTOFF_DISTANCE = 12


class ProteinFolder(torch.utils.data.Dataset):
    def __init__(self, folder, transforms=()):
        """Load graph samples in `*.pt` format from a folder
        :param folder: the dataset folder
        """
        folder = Path(folder).expanduser().resolve()
        if not folder.is_dir():
            raise ValueError(f'Not a directory: {folder}')

        self.transforms = transforms
        self.samples = sorted(f for f in folder.glob('sample*.pt'))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        path = self.samples[item]
        sample = torch.load(path)  # type: (str, str, tg.Graph, tg.Graph)

        for t in self.transforms:
            sample = t(*sample)

        return sample


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
    import tqdm
    import tables
    import pandas as pd

    destpath = Path(destpath).expanduser().resolve()
    destpath.mkdir(parents=True, exist_ok=True)

    local_lddt_scores = []
    global_lddt_scores = []
    global_gdtts_scores = []

    # First pass: enumerate proteins, count models, prepare frequency stats
    proteins = []
    total_models = 0
    h5_file = tables.open_file(filepath)
    for protein in h5_file.list_nodes('/'):
        # I'm not sure why the file contains these nodes, but they are not proteins for sure
        if protein._v_pathname.startswith(('/casp', '/cameo')):
            continue

        local_lddt_scores.append(np.array(protein.lddt).ravel())
        global_lddt_scores.append(np.array(protein.lddt_global))
        global_gdtts_scores.append(np.nanmean(protein.s_scores, axis=1))

        total_models += len(protein.names)
        proteins.append(protein)

    local_lddt_scores = pd.Series(np.concatenate(local_lddt_scores), name='local_lddt').dropna()
    global_lddt_scores = pd.Series(np.concatenate(global_lddt_scores), name='global_lddt').dropna()
    global_gdtts_scores = pd.Series(np.concatenate(global_gdtts_scores), name='global_gdtts').dropna()
    
    def make_frequency(series):
        """Quantize the values in a series into 20 equally spaced bins and compute frequency counts"""
        return pd.cut(series, bins=np.linspace(0, 1, num=20 + 1), include_lowest=True).value_counts().sort_index()

    # Compute weights for the local scores proportionally to the inverse of the frequency of
    # that score in the dataset, set minimum frequency to 1 to avoid +inf weights
    local_lddt_frequencies = make_frequency(local_lddt_scores)
    local_lddt_frequencies.to_pickle(destpath / 'local_lddt_frequencies.pkl')
    local_lddt_weights = local_lddt_frequencies.max() / local_lddt_frequencies.clip(lower=1)

    global_lddt_frequencies = make_frequency(global_lddt_scores)
    global_lddt_frequencies.to_pickle(destpath / 'global_lddt_frequencies.pkl')
    global_lddt_weights = global_lddt_frequencies.max() / global_lddt_frequencies.clip(lower=1)

    global_gdtts_frequencies = make_frequency(global_gdtts_scores)
    global_gdtts_frequencies.to_pickle(destpath / 'global_gdtts_frequencies.pkl')
    global_gdtts_weights = global_gdtts_frequencies.max() / global_gdtts_frequencies.clip(lower=1)

    # Second pass: save every model as a pytorch object
    protein_bar = tqdm.tqdm(proteins, desc='Proteins', unit=' proteins', leave=True)
    models_bar = tqdm.tqdm(total=total_models, desc='Models  ', unit=' models', leave=True)
    for i, protein in enumerate(protein_bar):
        for j in range(len(protein.names)):
            sample = protein_model_to_graph(protein, j, local_lddt_weights, global_lddt_weights, global_gdtts_weights)
            torch.save(sample, destpath / f'sample_{i}_{j}.pt')
            models_bar.update()
    protein_bar.close()
    models_bar.close()

    h5_file.close()


def protein_model_to_graph(protein, model_idx,
                           local_lddt_weights_table, global_lddt_weights_table, global_gdtts_weights_table):
    # Name of this protein
    protein_name = protein._v_pathname[1:]

    # Protein sequence and multi-sequence-alignment features, common to all models
    residues = protein.seq[0]                # One-hot encoding of the residues present in this model
    partial_entropy = protein.part_entr[0]   # Partial entropy of the residues
    self_information = protein.self_info[0]  # Self information of the residues
    sequence_length = len(residues)

    # Secondary structure is determined using some method by some research group (provider).
    # All models use the original sequence as source, but can fail to determine a structure for some residues.
    # The DSSP features are extracted for the residues whose 3D coordinates are determined.
    # Missing DSSP features are filled with 0s in the dataset.
    provider = protein.names[model_idx].decode('utf-8')    # Who built this model of the protein
    coordinates = protein.cb_coordinates[model_idx]        # Coordinates of the β carbon (α if β is not present)
    structure_determined = protein.valid_dssp[model_idx]   # Which residues are determined within this model
    native_determined = protein.valid_dssp[0]              # Which residues are determined in the native structure
    dssp_features = protein.dssp[model_idx]                # DSSP features for the secondary structure

    # For every residue, a score is determined by comparison with the native structure (using LDDT)
    # For some models, it is not possible to assign a score to a residue if:
    # - the experiment to determine the native model has failed to  _observe_ the secondary structure of that residue
    # - the algorithm to determine this model has failed to _determine_ the secondary structure of that residue
    # Frequency weights are looked up in the corresponding weights table (NaN scores get NaN weight).
    local_lddt = protein.lddt[model_idx]         # Quality scores of the residues
    local_lddt_valid = protein.valid[model_idx]  # Whether the local score is valid or not
    local_lddt[~local_lddt_valid] = np.nan       # For native structure the score is set to 1. for all residues,
                                                 # including non determined ones, we fix it by removing those scores.
    local_lddt_weights = np.full_like(local_lddt, fill_value=float('nan'))  # Frequency weights
    local_lddt_weights[local_lddt_valid] = local_lddt_weights_table.loc[local_lddt[local_lddt_valid]].values

    # The global LDDT score is an average of the local LDDT scores:
    # - residues missing from the model get a score of 0
    # - residues missing from the native structure are ignored in the average
    # Frequency weight us looked up in the corresponding weights table.
    global_lddt = protein.lddt_global[model_idx]                      # Quality score of the whole model
    global_lddt_weight = global_lddt_weights_table.loc[global_lddt]  # Frequency weight

    # The global GDT_TS score is another way of scoring the quality of a model.
    # Until we get GDT_TS in the dataset we use an average of the local S-score as a proxy for GDT_TS.
    # This is ok because GDT_TS and S-scores have a Pearson correlation of 1.
    # Since the S-score of missing residues is NaN, we ignore those scores when taking the mean.
    # Frequency weight is looked up in the corresponding weights table.
    local_sscore = protein.s_scores[model_idx]
    global_gdtts = np.nanmean(local_sscore)                             # Quality score of the whole model
    global_gdtts_weight = global_gdtts_weights_table.loc[global_gdtts]  # Frequency weight

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
            native_determined,
        ], axis=1)).float(),
        global_features=torch.tensor([
            global_lddt,
            global_gdtts,
            global_lddt_weight,
            global_gdtts_weight,
        ]).float()
    ).validate()

    return protein_name, provider, graph_in, graph_target


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
    to_fill = np.logical_and(separation == 0, np.isnan(distances))
    distances[to_fill] = np.random.normal(5.349724573740155, 0.9130922391969375, np.count_nonzero(to_fill))

    edge_features = np.concatenate((
        distances[:, None],
        separation_onehot
    ), axis=1)

    # Distances greater that 12 Angstrom are considered irrelevant and removed unless between adjacent residues.
    with np.errstate(invalid='ignore'):
        to_keep = np.logical_or(distances < MAX_CUTOFF_DISTANCE, separation == 0)

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
