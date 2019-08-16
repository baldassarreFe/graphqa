from pathlib import Path

import numpy as np
import torch.utils.data
import torchgraphs as tg

from . import features

MAX_DISTANCE = 12


class ProteinFolder(torch.utils.data.Dataset):
    def __init__(self, folder, cutoff):
        """
        Load `.pt` files from a folder
        :param folder: the dataset folder
        :param cutoff: the maximum distance at which non-adjacent residues should be connected,
                       passing 0 will result in simple linear graph structure
        """
        folder = Path(folder).expanduser().resolve()
        if not folder.is_dir():
            raise ValueError(f'Not a directory: {folder}')
        if cutoff > MAX_DISTANCE:
            from warnings import warn
            warn(f'The chosen cutoff {cutoff} is larger than the maximum distance saved in the dataset {MAX_DISTANCE}')

        self.cutoff = cutoff
        self.samples = sorted(f for f in folder.glob('sample*.pt'))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        path = self.samples[item]
        protein, provider, graph_in, graph_target = torch.load(path)  # type: (str, str, tg.Graph, tg.Graph)

        # Remove edges between non-adjacent residues with distance greater than cutoff
        edge_type = graph_in.edge_features[:, features.Input.Edge.EDGE_TYPE]
        distances = graph_in.edge_features[:, features.Input.Edge.SPATIAL_DISTANCE]
        to_keep = (edge_type == 1.) | (distances < self.cutoff)

        # Distances are encoded using a 0-centered RBF with variance proportional to the cutoff
        distances_rbf = torch.exp(- distances ** 2 / (2 * self.cutoff))

        graph_in = graph_in.evolve(
            senders=graph_in.senders[to_keep],
            receivers=graph_in.receivers[to_keep],
            edge_features=torch.stack([edge_type[to_keep], distances_rbf[to_keep]], dim=1),
        )

        return protein, provider, graph_in, graph_target


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
        global_gdtts_scores.append(np.array(protein.gdtts_global))

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
    # Since the S-score of missing residues is NaN, we consider those scores as 0s when taking the mean,
    # otherwise one could ignore them in the mean.
    # Frequency weight is looked up in the corresponding weights table.
    local_sscore = protein.s_scores[model_idx]
    global_gdtts = np.nan_to_num(local_sscore).mean()                   # Quality score of the whole model
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
    import scipy.spatial.distance

    # The distances are returned in a condensed upper triangular form without the diagonal
    distances = scipy.spatial.distance.pdist(coords)
    senders, receivers = np.triu_indices(len(coords), k=1)
    idx_adjacent = receivers - senders == 1  # indexes of adjacent residues

    # Chemically bonded adjacent residues = 1, non connected but close enough residues = 0
    edge_type = np.zeros_like(distances)
    edge_type[idx_adjacent] = 1.

    # The distance between adjacent residues might be missing because the residues don't have 3D coordinates
    # in this particular model, we set it to the a random distance with mean and variance equal to
    # the mean and variance of the distance between adjacent residues in the whole dataset
    to_fill = np.logical_and(idx_adjacent, np.isnan(distances))
    distances[to_fill] = np.random.normal(5.349724573740155, 0.9130922391969375, np.count_nonzero(to_fill))

    # Distances greater that 12 Angstrom are considered irrelevant and removed unless between adjacent residues
    with np.errstate(invalid='ignore'):
        is_relevant = distances < MAX_DISTANCE

    to_keep = np.logical_or(is_relevant, idx_adjacent)
    senders = senders[to_keep]
    receivers = receivers[to_keep]
    distances = distances[to_keep]
    edge_type = edge_type[to_keep]

    edge_features = np.vstack((edge_type, distances)).transpose()

    return senders, receivers, edge_features


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
