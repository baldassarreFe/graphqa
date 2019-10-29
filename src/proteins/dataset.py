import sys
from pathlib import Path
from typing import Union

import tqdm
import pyaml
import numpy as np
import pandas as pd
import torch.utils.data

from proteins.data import Decoy

MAX_CUTOFF_DISTANCE = 14


class ProteinQualityTarget(torch.utils.data.Dataset):
    def __init__(self, path: Union[str, Path], *,
                 node_features=('partial_entropy', 'self_information', 'dssp'),
                 cutoff=MAX_CUTOFF_DISTANCE):
        """
        :param node_features: which node features to load, all by default [partial_entropy, self_information, dssp]
        :param cutoff: the maximum distance at which non-adjacent residues should be connected,
                       passing 0 will result in simple linear graph structure
        """
        self.path = Path(path).expanduser().resolve()
        self.node_features = set(node_features)
        self.cutoff = cutoff
        self._target = None

        if self.cutoff > MAX_CUTOFF_DISTANCE:
            from warnings import warn
            warn(f'The chosen cutoff {cutoff} is larger than the maximum distance '
                 f'saved in the dataset {MAX_CUTOFF_DISTANCE}, this transformation will not remove any edge')

        with np.load(self.path, allow_pickle=True) as target:
            self.target_name = target['target_name'].item()
            self.num_decoys = len(target['decoy_names'])

    def load_eager(self):
        # Read the whole NpzFile into dict to avoid crc checks every time a field is accessed
        # If the dataloader uses forking, call this before forking to share the data at zero cost
        self._target = dict(np.load(self.path, allow_pickle=True))

    @property
    def target(self):
        # The file will be accessed lazily after forking to avoid concurrency issues
        if self._target is None:
            self._target = np.load(self.path, allow_pickle=True)
        return self._target

    def __len__(self):
        return self.num_decoys

    def __getitem__(self, decoy_idx):
        if len(self.node_features) > 0:
            node_features = []
            if 'partial_entropy' in self.node_features:
                node_features.append(self.target['partial_entropy'])
            if 'self_information' in self.node_features:
                node_features.append(self.target['self_information'])
            if 'dssp' in self.node_features:
                node_features.append(self.target['dssp'][decoy_idx])
            node_features = torch.from_numpy(np.concatenate(node_features, axis=1))
        else:
            node_features = torch.empty(len(self.target['residues']), 0)

        distances = self.target['distances'][decoy_idx]
        senders = self.target['senders'][decoy_idx]
        receivers = self.target['receivers'][decoy_idx]

        # Remove edges between non-adjacent residues with distance greater than cutoff
        if self.cutoff < MAX_CUTOFF_DISTANCE:
            is_peptide_bond = (receivers - senders) == 1
            to_keep = is_peptide_bond | (distances < self.cutoff)
            distances = distances[to_keep]
            senders = senders[to_keep]
            receivers = receivers[to_keep]

        decoy = Decoy(
            target_name=self.target['target_name'].item(),
            decoy_name=self.target['decoy_names'][decoy_idx],
            residues=torch.from_numpy(self.target['residues']),
            node_features=node_features,
            edge_features=torch.empty(len(senders), 0),
            distances=torch.from_numpy(distances),
            senders=torch.from_numpy(senders),
            receivers=torch.from_numpy(receivers),
            lddt=torch.from_numpy(self.target['lddt'][decoy_idx]),
            gdtts=torch.tensor([self.target['gdtts'][decoy_idx]]),
        )

        return decoy

    def __hash__(self):
        return hash(self.path)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.path.with_suffix("").name})'

    def __getstate__(self):
        # Not used: if the multiprocessing context is "fork" the dataloader is not pickled,
        # but shared with a copy-on-write mechanism
        print('getstate', {k for k, v in self.__dict__.items() if k != '_target'})
        return {k: v for k, v in self.__dict__.items() if k != '_target'}


def process_file(filepath: Union[str, Path], destpath: Union[str, Path], compress: bool):
    import tables

    filepath = Path(filepath).expanduser().resolve()
    h5_file = tables.open_file(filepath)

    destpath = Path(destpath).expanduser().resolve()
    destpath.mkdir(parents=True, exist_ok=False)

    scores = {
        'local': {
            'lddt': [],
            # 'sscore': [],
        },
        'global': {
            'gdtts': [],
            # 'gdtts_ha': [],
            # 'rmsd': [],
            # 'maxsub': [],
        }
    }

    sequence_lengths = []
    dataset_index = {
        'target': [],
        'decoy': [],
        'path': [],
        'index': [],
    }

    # First pass: enumerate proteins, count models, prepare frequency stats
    for protein in tqdm.tqdm(h5_file.list_nodes('/'), desc='Targets ', unit='t', leave=False):
        target_name = protein._v_name

        # I'm not sure why the file contains these nodes, but they are not proteins for sure
        if target_name.startswith(('casp', 'cameo')):
            continue

        if len(protein.names) <= 5:
            tqdm.tqdm.write(f'[{target_name}] Skipping target with {len(protein.names)} model(s).', file=sys.stderr)
            continue

        if len(protein.seq[0]) <= 50:
            tqdm.tqdm.write(f'[{target_name}] Skipping target with length {len(protein.seq[0])}.', file=sys.stderr)
            continue

        target = make_target(protein)
        if compress:
            np.savez_compressed(destpath / f'{target_name}.npz', **target)
        else:
            np.savez(destpath / f'{target_name}.npz', **target)
        with open(destpath / f'{target_name}_info.yaml', 'w') as f:
            pyaml.dump({
                'target': target_name,
                'num_residues': len(target['residues']),
                'num_decoys': len(target['decoy_names']),
            }, dst=f, sort_dicts=False)

        scores['local']['lddt'].append(target['lddt'].ravel())
        # scores['local']['sscore'].append(target['sscore'].ravel())
        scores['global']['gdtts'].append(target['gdtts'])
        # scores['global']['gdtts_ha'].append(target['gdtts_ha'])
        # scores['global']['rmsd'].append(target['rmsd'])
        # scores['global']['maxsub'].append(target['maxsub'])

        sequence_lengths.append(len(target['residues']))
        dataset_index['target'].extend([target['target_name']] * len(target['decoy_names']))
        dataset_index['decoy'].extend(target['decoy_names'])
        dataset_index['path'].extend([f'{target["target_name"]}.npz'] * len(target['decoy_names']))
        dataset_index['index'].extend(range(len(target['decoy_names'])))

    dataset_index = pd.DataFrame(dataset_index)
    dataset_index.to_csv(destpath / 'dataset_index.csv', header=True, index=False)

    # Bucketize the scores into 20 equally-spaced bins in the range [0, 1] and compute frequency counts for each bin
    bins = np.linspace(0, 1, num=20 + 1)
    for score_type in scores.keys():
        for score_name in scores[score_type].keys():
            scores[score_type][score_name] = pd.Series(np.concatenate(scores[score_type][score_name])).dropna()
            frequencies = pd.cut(
                scores[score_type][score_name], bins=bins, include_lowest=True).value_counts().sort_index()
            frequencies.to_pickle(destpath / f'{score_type}_{score_name}_frequencies.pkl')

    dataset_stats = {
        'dataset': Path(filepath).name,
        'num_targets': dataset_index['target'].nunique(),
        'num_decoys': len(dataset_index['decoy']),
        'avg_length': np.mean(sequence_lengths),
        'max_length': np.max(sequence_lengths),
        **{score_type: {
            score_name: scores[score_type][score_name].describe().to_dict()
            for score_name in scores[score_type]
        } for score_type in scores.keys()}
    }
    pyaml.print(dataset_stats, sort_dicts=False)
    with open(destpath / 'dataset_stats.yaml', 'w') as f:
        pyaml.dump(dataset_stats, dst=f, sort_dicts=False)

    h5_file.close()


def make_target(protein):
    # Name of this protein and of its decoys
    target_name = protein._v_name
    decoy_names = np.char.decode(protein.names, 'utf-8')  # D

    # Primary structure (sequence of amino acids) and multi-sequence-alignment features:
    # partial entropy and self information. These are common to all decoys.
    residues = np.argmax(protein.seq[0], axis=1)  # S       # TODO: 22 -> 20
    partial_entropy = protein.part_entr[0]        # S x 22
    self_information = protein.self_info[0]       # S x 22

    # Secondary structure is determined using some method by some research group.
    # All models use the original sequence as source, but can fail to determine a structure for some residues.
    # The DSSP features are extracted for the residues whose 3D coordinates are determined.
    # Missing DSSP features are filled with 0s in the dataset.
    dssp_features = np.concatenate((
        protein.dssp,                               # D x S x 14
        np.expand_dims(protein.valid_dssp, axis=2)  # D x S x  1
    ), axis=2)

    all_senders = []
    all_receivers = []
    all_distances = []
    for decoy_idx in range(len(decoy_names)):
        # Coordinates of the β carbon (α if β is not present) for this decoy
        coordinates = protein.cb_coordinates[decoy_idx]  # S x 3
        senders, receivers, distances = make_edges(coordinates)
        all_senders.append(senders)
        all_receivers.append(receivers)
        all_distances.append(distances)

    # For every residue, a score is determined by comparison with the native structure (using LDDT)
    # For some models, it is not possible to assign a score to a residue if:
    # - the experiment to determine the native model has failed to  _observe_ the secondary structure of that residue
    # - the algorithm to determine this model has failed to _determine_ the secondary structure of that residue
    local_lddt = np.array(protein.lddt)
    local_lddt_valid = np.array(protein.valid, dtype=np.bool)
    if np.any(local_lddt_valid != np.isfinite(local_lddt)):
        for invalid_decoy_idx in (local_lddt_valid != np.isfinite(local_lddt)).any(axis=1).nonzero()[0]:
            tqdm.tqdm.write(f'[{target_name}/{decoy_names[invalid_decoy_idx]}] Inconsistent LDDT: '
                            f'found {np.count_nonzero(np.isfinite(local_lddt[invalid_decoy_idx]))} not null scores, '
                            f'expected {np.count_nonzero(local_lddt_valid[invalid_decoy_idx])} valid scores.',
                            file=sys.stderr)
            if decoy_names[invalid_decoy_idx] == 'native':
                # Native models report a score of 1 for all residues, but some of them
                # were not determined in the experiment, so those should be marked invalid.
                local_lddt[invalid_decoy_idx, ~local_lddt_valid[invalid_decoy_idx]] = np.nan
            else:
                # Why does this happen? David says:
                # "It may be that the C alpha are present, so we can evaluate similarity,
                #  but not all the backbone is, so DSSP cannot compute the angles."
                local_lddt_valid[invalid_decoy_idx] = np.isfinite(local_lddt[invalid_decoy_idx])

            # Is this check even necessary?
            if np.isnan(local_lddt[invalid_decoy_idx, local_lddt_valid[invalid_decoy_idx]]).any():
                raise ValueError(
                    f'Found NaN values in local LDDT scores for {target_name}/{decoy_names[invalid_decoy_idx]}')

    # GDT_TS is a global score of the quality of a model.
    global_gdtts = np.array(protein.gdt_ts)
    if np.isnan(global_gdtts).any():
        for invalid_decoy_idx in np.isnan(global_gdtts).nonzero()[0]:
            raise ValueError(
                f'Found NaN values in global GDT_TS score for {target_name}/{decoy_names[invalid_decoy_idx]}')

    return {
        'target_name': target_name,
        'decoy_names': decoy_names,
        'residues': residues,
        'self_information': self_information,
        'partial_entropy': partial_entropy,
        'dssp': dssp_features,
        'senders': all_senders,
        'receivers': all_receivers,
        'distances': all_distances,
        'gdtts': global_gdtts,
        'lddt': local_lddt,
    }


def make_edges(coords):
    import scipy.spatial.distance

    # The distances are returned in a condensed upper triangular form without the diagonal,
    # e.g. for 5 coordinates, we would have:
    # distances = [dist(p[0], p[1]), dist(p[0], p[2]), dist(p[0], p[3]), ..., dist(p[3], p[4])]
    # senders   = [0, 0, 0, 0, 1, 1, 1, 2, 2, 3]
    # receivers = [1, 2, 3, 4, 2, 3, 4, 3, 4, 4]
    distances = scipy.spatial.distance.pdist(coords)
    senders, receivers = np.triu_indices(len(coords), k=1)
    separation = receivers - senders - 1

    # The spatial distance between adjacent residues might be missing (NaN) because the residues don't have
    # 3D coordinates in this particular model. For adjacent residues (separation = 0) we set the edge length
    # to a random value with mean and variance equal to the mean and variance of the distance between
    # adjacent residues in the whole dataset. For residues further apart (separation > 0) we leave NaN.
    to_fill = (separation == 0) & (np.isnan(distances))
    distances[to_fill] = np.random.normal(5.349724573740155, 0.9130922391969375, np.count_nonzero(to_fill))

    # Distances > MAX_CUTOFF_DISTANCE Angstrom are considered irrelevant and removed unless between adjacent residues.
    with np.errstate(invalid='ignore'):
        to_keep = (distances < MAX_CUTOFF_DISTANCE) | (separation == 0)

    return senders[to_keep], receivers[to_keep], distances[to_keep].astype(np.float32)


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
    sp_preprocess.add_argument('--compress', action='store_true')
    sp_preprocess.set_defaults(command=process_file)

    sp_describe = subparsers.add_parser('describe', help='Describe existing datasets')
    sp_describe.add_argument('--filepath', type=str, required=True)
    sp_describe.set_defaults(command=describe)

    args = parser.parse_args()
    args.command(**{k: v for k, v in vars(args).items() if k != 'command'})


if __name__ == '__main__':
    main()
