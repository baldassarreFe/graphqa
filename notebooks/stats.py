import tables
import tqdm
import numpy as np
import scipy.spatial
from proteins.utils import RunningStats

h5_file = tables.open_file('../data/training_casp9_10.v4.h5')

all_distances = RunningStats()
all_neighbor_distances = RunningStats()
for protein in tqdm.tqdm(h5_file.list_nodes('/')):
    for model_idx in tqdm.trange(len(protein.names)):
        residue_mask = protein.valid_dssp[model_idx]                   # which residues are actually present in this model
        coordinates = protein.cb_coordinates[model_idx][residue_mask]  # coordinates of the β carbon (or α if β is not present)
        distances = scipy.spatial.distance.pdist(coordinates)          # flattened pairwise distances between residues
        all_distances.add_from(distances)
        neighbor_distances = np.sqrt(np.square(coordinates[:-1] - coordinates[1:]).sum(axis=1))
        all_neighbor_distances.add_from(neighbor_distances)
print(all_distances)
print(all_neighbor_distances)