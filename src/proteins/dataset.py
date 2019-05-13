import scipy.spatial
import tables
import torch.utils.data
import numpy as np
import torchgraphs as tg

class ProteinModels(torch.utils.data.Dataset):
    def __init__(self, protein):
        self.protein = protein
        self.name = protein._v_pathname[1:]
        self.num_models = len(protein.names)

    def __len__(self):
        return self.num_models

    def __getitem__(self, model_idx):
        provider = self.protein.names[model_idx].decode('utf-8') # who built this model of the protein

        residue_mask = self.protein.valid_dssp[model_idx]                # which residues are actually present in this model
        residues = self.protein.seq[0][residue_mask]                     # one-hot encoding of the residues present in this model
        partial_entropy = self.protein.part_entr[0][residue_mask]        # partial entropy of the residues
        self_information = self.protein.self_info[0][residue_mask]       # self information of the residues
        secondary_structure = self.protein.dssp[model_idx][residue_mask] # secondary structure features

        coordinates = self.protein.cb_coordinates[model_idx][residue_mask] # coordinates of the β carbon (or α if β is not present)
        distances = scipy.spatial.distance.pdist(coordinates)              # pairwise distances between residues

        valid_scores = self.protein.valid[model_idx][residue_mask] # of the residues present in this model, only those that are also present in the experimental model have a score
        scores = self.protein.lddt[model_idx][residue_mask]        # quality scores of the residues
        global_score = self.protein.lddt_global[model_idx]         # quality score of the whole model

        N = len(residues)

        distances_idx = distances < 10
        senders, receivers = np.triu_indices(N, k=1)
        senders = senders[distances_idx]
        receivers = receivers[distances_idx]

        graph_in = tg.Graph(
            num_nodes=N,
            node_features=torch.from_numpy(np.concatenate([
                #residues,
                partial_entropy,
                self_information,
                secondary_structure
            ], axis=1)).float(),
            num_edges=2 * np.count_nonzero(distances_idx),
            senders=torch.from_numpy(np.concatenate((senders, receivers))),
            receivers=torch.from_numpy(np.concatenate((receivers, senders))),
            edge_features=torch.from_numpy(- (distances[distances_idx] - 3) / 3).exp_().repeat(2).view(-1, 1).float()
        )
        graph_target = tg.Graph(
            num_nodes=N,
            node_features=torch.from_numpy(scores).view(-1, 1).float(),
            global_features=torch.tensor([global_score]).float()
        )

        return self.name, provider, graph_in, graph_target


class ProteinFile(torch.utils.data.ConcatDataset):
    def __init__(self, file):
        super(ProteinFile, self).__init__([
            ProteinModels(protein)
            for protein in tables.open_file(file).list_nodes('/')
        ])
