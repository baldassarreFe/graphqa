import argparse
import shutil
import tempfile
import subprocess
import multiprocessing
from pathlib import Path
from typing import Union

import Bio.AlignIO
import Bio.Align
import Bio.SeqIO
import Bio.SeqRecord
import numpy as np
import pandas as pd
from loguru import logger

from .aminoacids import aa_1_mapping, aa_1_mapping_inv

msa_bins = np.arange(22)
msa_1_mapping = {**aa_1_mapping, "X": 21, "-": 22}
msa_1_mapping_inv = aa_1_mapping_inv + ["X"]


class MultipleSequenceAlignment(object):
    def __init__(self, database_path: Union[str, Path], cpus=None):
        """
        Args:
            database_path: path to protein database
        """
        if cpus is None:
            cpus = max(1, multiprocessing.cpu_count() - 1)
        self.cpus = cpus
        self.database_path = database_path.expanduser().resolve()

    def run_on_file(self, input_path: Path, output_path: Path):
        # TODO consider making output_path optional,
        #      if present then use subprocess.run and let jackhmmer write to that file,
        #      if absent use something like:
        #      process = subprocess.Popen(command, stdout=subprocess.PIPE)
        #      yield from Bio.AlignIO.parse(process.stdout, format="stockholm")
        #      exitcode = process.poll()
        with tempfile.TemporaryDirectory(prefix="jackhmmer_") as tmpdir:
            tmpdir = Path(tmpdir)
            # stdout could be very large, better to let jackhmmer write to a file
            stdout_path = tmpdir / "jackhmmer.out"
            alignment_path = tmpdir / "jackhmmer.sto"

            result = subprocess.run(
                [
                    "jackhmmer",
                    "-o",
                    stdout_path.as_posix(),
                    "-A",
                    alignment_path.as_posix(),
                    "-N",
                    "3",
                    "-E",
                    ".001",
                    "--cpu",
                    str(self.cpus),
                    input_path.as_posix(),
                    self.database_path.as_posix(),
                ],
                check=False,
            )
            if result.returncode != 0:
                with stdout_path.open() as f:
                    stdout = "".join(f.readlines())
                    raise JackhmmerError(result.returncode, stdout)
            else:
                alignment_path.parent.mkdir(exist_ok=True, parents=True)
                shutil.move(alignment_path, output_path)

    def run_on_sequence(
        self, sequence: Bio.SeqRecord.SeqRecord
    ) -> Bio.Align.MultipleSeqAlignment:
        with tempfile.TemporaryDirectory(prefix="jackhmmer_") as tmpdir:
            tmpdir = Path(tmpdir)
            tmp_fasta = tmpdir / "sequence.fasta"
            tmp_alignment = tmpdir / "sequence.sto"

            with tmp_fasta.open("w") as f:
                Bio.SeqIO.write(sequence, f, format="fasta")
            self.run_on_file(tmp_fasta, tmp_alignment)

            with tmp_alignment.open() as f:
                alignment = next(Bio.AlignIO.parse(f, format="stockholm"))
            return alignment


class JackhmmerError(Exception):
    def __init__(self, exit_code: int, error_message: str):
        self.exit_code = exit_code
        self.error_message = error_message


def compute_counts(alignment: Bio.Align.MultipleSeqAlignment) -> pd.DataFrame:
    original_sequence = [aa for aa in alignment[0] if aa != "-"]
    seq_length = len(original_sequence)
    msa_counts = np.empty((seq_length, 21), dtype=np.int)

    idx_in_seq = 0
    for idx_in_msa, aa in enumerate(alignment[0]):
        if aa == "-":
            continue
        msa_at_idx = [msa_1_mapping[seq[idx_in_msa]] for seq in alignment]
        counts = np.histogram(msa_at_idx, bins=msa_bins)[0]
        msa_counts[idx_in_seq] = counts
        idx_in_seq += 1

    msa_counts = pd.DataFrame(
        msa_counts,
        index=pd.Index(original_sequence, name="aa"),
        columns=msa_1_mapping_inv,
    )
    return msa_counts


def compute_msa_features(msa_counts: pd.DataFrame) -> pd.DataFrame:
    msa_freq = msa_counts / msa_counts.values.sum(axis=1, keepdims=True)
    return msa_freq


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Jackhmmer over all sequences in a fasta file."
    )
    parser.add_argument("input", help="Primary sequences.")
    parser.add_argument("database", help="Database of sequences.")
    parser.add_argument("output", help="Output path (pickled MSA counts)")
    parser.add_argument("--cpus", type=int, required=False, help="Number of cpus")
    # parser.add_argument("--verbose", "-v", action="count", default=0)
    # TODO add argument --skip-existing
    args = parser.parse_args()
    return args


def run_msa(input, database, output, cpus):
    input = Path(input).expanduser().resolve()
    database = Path(database).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    hmmer_out = output.parent.joinpath("jackhmmer.sto")

    with input.open() as f:
        num_sequences = sum((1 for s in Bio.SeqIO.parse(f, "fasta")), 0)
    logger.info(f"Running on {num_sequences} sequence(s)")

    msa = MultipleSequenceAlignment(database, cpus)
    msa.run_on_file(input, hmmer_out)
    with hmmer_out.open() as f:
        msa_dict = {
            alignment[0].name: compute_counts(alignment)
            for alignment in Bio.AlignIO.parse(f, format="stockholm")
        }
        pd.to_pickle(msa_dict, output)


def main():
    args = parse_args()
    run_msa(**vars(args))


"""
python -m graphqa.data.msa input.fasta database.fasta output.sto
"""
if __name__ == "__main__":
    main()
