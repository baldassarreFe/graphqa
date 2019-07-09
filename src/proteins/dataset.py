from pathlib import Path

import numpy as np
import torch.utils.data

import torchgraphs as tg

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
        protein_name, provider, graph_in, graph_target = torch.load(self.samples[item])

        distances, edge_type = graph_in.edge_features.unbind(dim=1)
        to_keep = (edge_type == 1.) | (distances < self.cutoff)
        edge_features = graph_in.edge_features[to_keep]
        edge_features[:, 0] = (- edge_features[:, 0].pow(2) / (2 * self.cutoff)).exp()

        graph_in = graph_in.evolve(
            num_edges=len(edge_features),
            senders=graph_in.senders[to_keep],
            receivers=graph_in.receivers[to_keep],
            edge_features=edge_features,
        )

        return protein_name, provider, graph_in, graph_target


def process_file(filepath, destpath):
    import tables

    destpath = Path(destpath).expanduser().resolve()
    destpath.mkdir(parents=True, exist_ok=True)

    with tables.open_file(filepath) as h5_file:
        proteins = (
            p for p in h5_file.list_nodes('/')
            if not (p._v_pathname.startswith('/casp') or p._v_pathname.startswith('/cameo'))
        )
        for i, protein in enumerate(proteins):
            num_models = len(protein.names)
            for j in range(num_models):
                sample = process_protein(protein, j)
                torch.save(sample, destpath / f'sample_{i}_{j}.pt')


def process_protein(protein, model_idx):
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
    dssp_features = protein.dssp[model_idx]                # DSSP features for the secondary structure

    # For every residue a score is determined by comparison with the native structure.
    # The global score is an average of the local scores.
    # It is not possible to assign a score to a residue if:
    # - the native model of the protein determined experimentally has failed to
    #   _observe_ the secondary structure of that residue
    # - the current model of the protein has failed to _determine_ the secondary structure of that residue
    scores = protein.lddt[model_idx]               # Quality scores of the residues
    global_score = protein.lddt_global[model_idx]  # Quality score of the whole model
    # valid_scores = protein.valid[model_idx]      # Whether the local score is valid or not

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
        num_edges=2 * len(edge_features),
        senders=torch.from_numpy(np.concatenate((senders, receivers))),
        receivers=torch.from_numpy(np.concatenate((receivers, senders))),
        edge_features=torch.from_numpy(edge_features).repeat(2, 1).float()
    )
    graph_target = tg.Graph(
        num_nodes=sequence_length,
        node_features=torch.from_numpy(scores).view(-1, 1).float(),
        global_features=torch.tensor([global_score]).float()
    )

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

    edge_features = np.vstack((distances, edge_type)).transpose()

    return senders, receivers, edge_features


def describe(filepath):
    import tables
    import tqdm
    import numpy as np
    import scipy.spatial
    from proteins.utils import RunningStats

    h5_file = tables.open_file(filepath)

    all_distances = RunningStats()
    all_neighbor_distances = RunningStats()
    for protein in tqdm.tqdm(h5_file.list_nodes('/')):
        for model_idx in tqdm.trange(len(protein.names)):
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
