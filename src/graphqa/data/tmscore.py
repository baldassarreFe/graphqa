from __future__ import annotations

from loguru import logger

"""
First download and compile the executable from [Zhang lab](https://zhanglab.ccmb.med.umich.edu/TM-score/):

```bash
wget -q 'https://zhanglab.ccmb.med.umich.edu/TM-score/TMscore.cpp' &&
g++ -static -O3 -ffast-math -lm -o TMscore TMscore.cpp &&
rm TMscore.cpp
```

Tested with latest update 2019/11/25 (see .cpp file).

Run as:
python -m graphqa.data.tmscore native.pdb decoys/ output.npz
"""
import shutil
import argparse
import tempfile
import contextlib
import subprocess
from pathlib import Path
from typing import Union, NamedTuple, List, Iterator

import numpy as np


class TmScore(object):
    def __init__(self, tm_score="TMscore", timeout_sec=90):
        self.tm_score = shutil.which(tm_score)
        if self.tm_score is None:
            raise ValueError(f"TMscore executable not found: {tm_score}")
        self.timeout_sec = timeout_sec

    @staticmethod
    def _decode_stdout(
        e: Union[subprocess.TimeoutExpired, subprocess.CalledProcessError]
    ):
        try:
            return e.stderr.decode()
        except (AttributeError, UnicodeDecodeError):
            return "<no stderr>"

    def score_single(self, native_path: str, decoy_path: str):
        """

        Args:
            native_path:
            decoy_path:

        Returns:

        """

        try:
            result = subprocess.run(
                [
                    self.tm_score,
                    Path(native_path).expanduser().resolve().as_posix(),
                    Path(decoy_path).expanduser().resolve().as_posix(),
                    "-outfmt",
                    "-1",
                ],
                capture_output=True,
                check=True,
                timeout=self.timeout_sec,
            )
        except subprocess.TimeoutExpired as e:
            msg = self._decode_stdout(e)
            raise TmScoreError(f"Timed out {native_path} {decoy_path}: {msg}")
        except subprocess.CalledProcessError as e:
            msg = self._decode_stdout(e)
            raise TmScoreError(
                f"Exit code {e.returncode} {native_path} {decoy_path}: {msg}"
            )

        stdout = iter(result.stdout.decode().splitlines())
        scores = self._parse_single(stdout)
        return scores

    def score_dir(self, native_path: str, decoys_dir: str) -> List[TmScoreOutput]:
        """

        Args:
            native_path:
            decoys_dir:

        Returns:

        """
        with self._decoys_list_file(decoys_dir) as (decoys_list, num_decoys):
            try:
                result = subprocess.run(
                    [
                        self.tm_score,
                        "-dir1",
                        # TMscore really requires the dir to have a trailing '/'
                        Path(decoys_dir).expanduser().resolve().as_posix() + '/',
                        decoys_list,
                        Path(native_path).expanduser().resolve().as_posix(),
                        "-outfmt",
                        "-1",
                    ],
                    capture_output=True,
                    check=True,
                    timeout=self.timeout_sec,
                )
            except subprocess.TimeoutExpired as e:
                msg = self._decode_stdout(e)
                raise TmScoreError(f"Timed out {native_path} {decoys_dir}: {msg}")
            except subprocess.CalledProcessError as e:
                msg = self._decode_stdout(e)
                raise TmScoreError(
                    f"Exit code {e.returncode} {native_path} {decoys_dir}: {msg}"
                )

        scores = []
        stdout = iter(result.stdout.decode().splitlines())
        for i in range(num_decoys):
            scores.append(self._parse_single(stdout))
        return scores

    @classmethod
    def _parse_single(cls, lines: Iterator[str]):
        def parse(pattern, action):
            # Advance and extract
            for line in lines:
                if line.startswith(pattern):
                    return action(line)
            raise ValueError(f"Not found: {pattern}")

        decoy = parse("Structure1:", lambda l: l.split()[1][:-4])
        rmsd = parse("RMSD of  the common residues=", lambda l: float(l.split()[-1]))
        tmscore = parse("TM-score    =", lambda l: float(l.split()[2]))
        gdt_ts = parse("GDT-TS-score=", lambda l: float(l.split()[1]))
        gdt_ha = parse("GDT-HA-score=", lambda l: float(l.split()[1]))

        return TmScoreOutput(decoy, rmsd, tmscore, gdt_ts, gdt_ha)

    @staticmethod
    @contextlib.contextmanager
    def _decoys_list_file(decoys_dir):
        # TMscore can score all decoys in a directory by reading
        # their paths one per line from a text file
        decoys_dir = Path(decoys_dir).expanduser().resolve()
        if not decoys_dir.is_dir():
            raise ValueError(f'Not a directory: {decoys_dir}')

        decoys = list(decoys_dir.glob("*.pdb"))
        if len(decoys) == 0:
            logger.warning(f"No decoys foud in {decoys_dir}")

        with tempfile.NamedTemporaryFile("w", prefix='tmscore', suffix='.txt') as f:
            f.writelines(f"{p.name}\n" for p in decoys)
            f.flush()
            yield f.name, len(decoys)


class TmScoreOutput(NamedTuple):
    decoy: str
    rmsd: float
    tmscore: float
    gdt_ts: float
    gdt_ha: float


class TmScoreError(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("native_pdb")
    parser.add_argument("decoys_dir")
    parser.add_argument("output_npz", help="Output path (numpy archive format)")
    parser.add_argument("--tmscore", default="TMscore", help="TMscore executable")
    parser.add_argument("--timeout", default=90, type=int, help="TMscore timeout")
    args = parser.parse_args()

    tmscore = TmScore(args.tmscore, timeout_sec=args.timeout)
    scores = tmscore.score_dir(args.native_pdb, args.decoys_dir)
    # Dict of lists from list of named tuples (ugly)
    scores_dict = dict(zip(TmScoreOutput._fields, zip(*scores)))
    scores_dict = {k: np.array(v) for k, v in scores_dict.items()}
    np.savez(args.output_npz, **scores_dict)


if __name__ == "__main__":
    main()
