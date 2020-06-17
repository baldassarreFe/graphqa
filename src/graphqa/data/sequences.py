from pathlib import Path

import Bio.SeqIO
import numpy as np

from .aminoacids import aa_1_mapping


class FastaToNumpyWrapper(object):
    """
    Wrap a fasta file containing sequences, allow random access using a cache, encode sequences as uint8 arrays.
    """

    def __init__(self, path):
        path = Path(path).expanduser().resolve().as_posix()
        self.seq_iterator = Bio.SeqIO.parse(path, format="fasta")
        self._cache = {}

    @staticmethod
    def encode_seq(seq):
        return np.array([aa_1_mapping[aa] for aa in seq], dtype=np.uint8)

    def __getitem__(self, item):
        if item not in self._cache:
            for seq in self.seq_iterator:
                self._cache[seq.name] = self.encode_seq(seq)
                if seq.name == item:
                    break
        return self._cache.pop(item)

    def __str__(self):
        return f"{self.__class__.__name__}({len(self._cache)} cached sequences)"