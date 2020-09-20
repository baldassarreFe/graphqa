"""
Compute CAD scores between a native structure and a folder of decoys using Voronota.

First download Voronota:
```bash
wget -q 'https://github.com/kliment-olechnovic/voronota/releases/download/v1.21.2744/voronota_1.21.2744.tar.gz'
tar xzf voronota_1.21.2744.tar.gz
```

Run from the command line as:
```bash
python -m graphqa.data.cadscore -vv \
  --voronota voronota_1.21.2744/voronota-cadscore
  CASP9/native/T0522.pdb \
  CASP9/decoys/T0522/ \
  CASP9/decoys/T0522.cad.npz \
  134
```
"""

import argparse
import contextlib
import tempfile
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from loguru import logger
import sys
import time

TIMEOUT_SEC = 90
logger.disable("graphqa.data.cadscore")


class CadScoreError(RuntimeError):
    pass


@contextlib.contextmanager
def decoys_list_file(decoys_dir):
    decoys = Path(decoys_dir).glob("*.pdb")

    with tempfile.NamedTemporaryFile("w") as tmp_file:
        tmp_file.writelines(f"{p.name}\n" for p in decoys)
        tmp_file.flush()

        yield tmp_file.name


def parse_output(lines):
    result = {"decoys": [], "rmsd": [], "tm_score": [], "gdt_ts": [], "gdt_ha": []}

    for line in lines:
        if line.startswith("Structure1:"):
            result["decoys"].append(line.split()[1])
        elif line.startswith("RMSD of  the common residues="):
            result["rmsd"].append(float(line.split()[-1]))
        elif line.startswith("TM-score    ="):
            result["tm_score"].append(float(line.split()[2]))
        elif line.startswith("GDT-TS-score="):
            result["gdt_ts"].append(float(line.split()[1]))
        elif line.startswith("GDT-HA-score="):
            result["gdt_ha"].append(float(line.split()[1]))

    return {k: np.array(v) for k, v in result.items()}


def compute_scores(
    native_path: str,
    decoys_dir: str,
    sequence_length: int,
    voronota="voronota-cadscore",
):
    residue_index = pd.RangeIndex(sequence_length, name="residue_idx")
    decoys = []
    local_scores = []
    global_scores = []

    with tempfile.TemporaryDirectory() as tmpdir:
        decoy_paths = list(Path(decoys_dir).glob("*.pdb"))
        logger.info(f"Running CAD score on {len(decoy_paths)} decoys")
        start = time.time()
        for decoy_path in decoy_paths:
            logger.debug(decoy_path)
            decoy_name = decoy_path.with_suffix("").name
            cad_scores_path = Path(tmpdir) / decoy_name
            try:
                result = subprocess.run(
                    [
                        voronota,
                        "--input-target",
                        Path(native_path).expanduser().resolve().as_posix(),
                        "--input-model",
                        decoy_path.expanduser().resolve().as_posix(),
                        "--output-residue-scores",
                        cad_scores_path.expanduser().resolve().as_posix(),
                        "--cache-dir",
                        tmpdir,
                        "--contacts-query-by-code",
                        "AS",
                    ],
                    capture_output=True,
                    check=True,
                    timeout=TIMEOUT_SEC,
                )
            except subprocess.TimeoutExpired as e:
                try:
                    msg = e.stderr.decode()
                except:
                    msg = "<no stderr>"
                logger.warning(f"Timed out {decoy_path}: {msg}")
                continue
            except subprocess.CalledProcessError as e:
                try:
                    msg = e.stderr.decode()
                except:
                    msg = "<no stderr>"
                logger.warning(f"Exit code {e.returncode} {decoy_path}: {msg}")
                continue

            try:
                # Parse local scores from output file
                df = pd.read_csv(
                    cad_scores_path, delimiter=" ", names=["residue_str", "local_cad"]
                )
                df.index = (
                    df["residue_str"]
                    .str.extract("r<(\d+)>", expand=False)
                    .astype(int)
                    .values
                    - 1
                )
                local_cad = (
                    df["local_cad"].reindex(residue_index, fill_value=np.nan).values
                )
                # Parse global score from stdout
                global_score = float(result.stdout.decode().split()[4])
            except FileNotFoundError as e:
                logger.warning(f"CAD score did not produce local residue output")
            except Exception as e:
                logger.warning(f"Error while parsing output: {e}")
            else:
                decoys.append(decoy_name)
                local_scores.append(local_cad)
                global_scores.append(global_score)

    logger.info(
        f"Done {len(decoys)} out of {len(decoy_paths)} in {time.time() - start:.1f} seconds"
    )

    if len(decoys) == 0 or len(local_scores) == 0 or len(global_scores) == 0:
        raise CadScoreError(f"No decoy was successfully evaluated for {native_path}")

    return {
        "decoys": np.array(decoys),
        "local_cad": np.stack(local_scores, axis=0),
        "global_cad": np.array(global_scores),
    }


class LogFilter(object):
    _mapping = {0: "WARNING", 1: "INFO", 2: "DEBUG", 3: "TRACE"}

    def __init__(self, verbose):
        level = LogFilter._mapping.get(verbose, "TRACE")
        self.level_no = logger.level(level).no

    def __call__(self, record):
        return record["level"].no >= self.level_no


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("native_pdb")
    parser.add_argument("decoys_dir")
    parser.add_argument("output_npz", help="Output path (numpy archive format)")
    parser.add_argument("sequence_length", type=int)
    parser.add_argument(
        "--voronota",
        default="voronota-cadscore",
        help="Path to voronota-cadscore executable if not in your PATH",
    )
    parser.add_argument("--verbose", "-v", action="count", default=0)
    args = parser.parse_args()
    return args


def run_cadscore(native_pdb, decoys_dir, output_npz, sequence_length, voronota):
    scores = compute_scores(native_pdb, decoys_dir, sequence_length, voronota)
    np.savez(output_npz, **scores)


def main():
    args = vars(parse_args())

    logger.enable("graphqa.data.cadscore")
    logger.remove()
    logger.add(sys.stderr, filter=LogFilter(args.pop("verbose")), level=0)

    run_cadscore(**args)


if __name__ == "__main__":
    main()
