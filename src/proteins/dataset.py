import tables
import numpy as np
import torch.utils.data
import scipy.spatial.distance

import torchgraphs as tg

# TODO add missing residues, add sequence info

class ProteinModels(torch.utils.data.Dataset):
    def __init__(self, filepath, protein_name):
        self.filepath = filepath
        self.protein_name = protein_name
        with tables.open_file(self.filepath) as h5_file:
            self.num_models = len(h5_file.get_node(f'/{self.protein_name}').names)

    def __len__(self):
        return self.num_models

    def __getitem__(self, model_idx):
        with tables.open_file(self.filepath) as h5_file:
            protein = h5_file.get_node(f'/{self.protein_name}')

            # Protein sequence and multi-sequence-alignment features, common to all models
            residues = protein.seq[0]                # One-hot encoding of the residues present in this model
            partial_entropy = protein.part_entr[0]   # Partial entropy of the residues
            self_information = protein.self_info[0]  # Self information of the residues
            sequence_length = len(residues)

            # Secondary structure is determined using some method by some research group.
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

            senders, receivers, distances = ProteinModels.make_edges(coordinates)

            graph_in = tg.Graph(
                num_nodes=sequence_length,
                node_features=torch.from_numpy(np.concatenate([
                    residues,
                    partial_entropy,
                    self_information,
                    dssp_features,
                    structure_determined[:, None] * 2 - 1
                ], axis=1)).float(),
                num_edges=2 * len(distances),
                senders=torch.from_numpy(np.concatenate((senders, receivers))),
                receivers=torch.from_numpy(np.concatenate((receivers, senders))),
                edge_features=torch.from_numpy(distances).repeat(2).view(-1, 1).float()
            )
            graph_target = tg.Graph(
                num_nodes=sequence_length,
                node_features=torch.from_numpy(scores).view(-1, 1).float(),
                global_features=torch.tensor([global_score]).float()
            )

            return self.protein_name[1:], provider, graph_in, graph_target

    @staticmethod
    def make_edges(coords):
        distances = scipy.spatial.distance.pdist(coords)
        senders, receivers = np.triu_indices(len(coords), k=1)

        # The distance between adjacent residues might be missing because the residues don't have 3D coordinates,
        # we set it to the average distance between adjacent residues
        to_fill = np.logical_and(receivers - senders == 1, np.isnan(distances))
        distances[to_fill] = np.random.normal(5.349724573740155, 0.9130922391969375, np.count_nonzero(to_fill))

        # Distances over 10 Angstrom are considered irrelevant
        with np.errstate(invalid='ignore'):
            is_relevant = distances < 10

        senders = senders[is_relevant]
        receivers = receivers[is_relevant]
        distances = distances[is_relevant]

        # f(5) = 1.0, f(10) = .01
        distances = np.exp((distances - 5) * np.log(.01) / 5)

        return senders, receivers, distances


class ProteinFile(torch.utils.data.ConcatDataset):
    def __init__(self, filepath):
        with tables.open_file(filepath) as h5_file:
            super(ProteinFile, self).__init__([
                ProteinModels(filepath, protein._v_pathname[1:])
                for protein in h5_file.list_nodes('/') if not protein._v_pathname.startswith('/casp')
            ])

