import argparse
import contextlib
import os
import tempfile
import warnings
from pathlib import Path
from typing import overload

import Bio.Align.AlignInfo
import Bio.AlignIO
import Bio.PDB
import Bio.PDB.Structure
import Bio.SeqIO
import docker
import docker.types
from joblib import Parallel, delayed
from loguru import logger

from .decoys import parse_pdb

from graphqa.utils import extract_tar


class DsspDocker(object):
    def __init__(self, input_dir=None, output_dir=None, skip_existing=False):
        self.input_dir = None
        if input_dir is not None:
            self.input_dir = Path(input_dir).expanduser().resolve()

        self.output_dir = None
        if output_dir is not None:
            self.output_dir = Path(output_dir).expanduser().resolve()

        self.client = docker.from_env()
        self.tmp_dir = None
        self.container = None
        self.image = self.client.images.get("dssp")

        self.skip_existing = skip_existing

    def start(self):
        mounts = []
        if self.input_dir is not None:
            if not self.input_dir.is_dir():
                raise ValueError(f"Not a directory: {self.input_dir}")
            mounts.append(
                docker.types.Mount(
                    source=self.input_dir.as_posix(),
                    target="/input",
                    type="bind",
                    read_only=True,
                )
            )

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tmp_dir = tempfile.TemporaryDirectory(suffix="_docker_dssp")
        mounts.append(
            docker.types.Mount(
                source=self.tmp_dir.name, target="/tmp", type="bind", read_only=False
            )
        )

        self.container = self.client.containers.run(
            image=self.image,
            user=os.getuid(),
            tty=True,
            detach=True,
            stdout=False,
            stderr=False,
            auto_remove=True,
            mounts=mounts,
        )

    def stop(self):
        if self.container:
            self.container.stop(timeout=10)
        if self.tmp_dir:
            self.tmp_dir.cleanup()

    @overload
    def run(self, path: Path) -> Path:
        ...

    @overload
    def run(self, structure: Bio.PDB.Structure.Structure) -> Bio.PDB.DSSP:
        ...

    def run(self, input):
        if not self.container:
            raise RuntimeError(
                "DSSP container is not running, call start() first "
                "or use this instance as context manager."
            )

        if isinstance(input, Bio.PDB.Structure.Structure):
            return self._run_on_structure(input)

        if isinstance(input, Path):
            input = input.expanduser().resolve()
            if not input.is_file():
                raise FileNotFoundError(input)
            return self._run_on_file(input)

        raise ValueError(input)

    def _run_on_file(self, path_outside: Path):
        if self.input_dir is None:
            raise ValueError("Can only run on .pdb files if `input_dir` is specified")
        if self.output_dir is None:
            raise ValueError("Can only run on .pdb files if `output_dir` is specified")

        dest = path_outside.relative_to(self.input_dir).with_suffix(".dssp")
        dest = self.output_dir / dest
        if self.skip_existing and dest.is_file():
            return dest

        try:
            path_inside = Path("/input") / path_outside.relative_to(self.input_dir)
            stdout = self._docker_dssp(path_inside)
        except DsspError as e:
            # DSSP has often troubles with .pdb filws with lines shorter than 80 chars.
            # Retry with a reformatted version of the structure.
            structure = parse_pdb(path_outside)
            with self._tempfile_pdb(structure) as path_inside:
                stdout = self._docker_dssp(path_inside)

        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            f.write(stdout)
        return dest

    def _run_on_structure(self, structure: Bio.PDB.Structure.Structure):
        """

        Args:
            structure: will be modified in place

        Returns:

        """
        with self._tempfile_pdb(structure) as path_inside:
            stdout = self._docker_dssp(path_inside)

        with tempfile.NamedTemporaryFile(mode="wb") as f:
            f.write(stdout)
            f.flush()
            dssp = Bio.PDB.DSSP(structure[0], in_file=f.name, file_type="DSSP")

        return dssp

    def _docker_dssp(self, path_inside: Path):
        exit_code, (stdout, stderr) = self.container.exec_run(
            cmd=["/app/mkdssp", path_inside.as_posix()], demux=True
        )

        if exit_code != 0:
            stderr = stderr.decode().strip() if stderr is not None else "no stderr"
            raise DsspError(exit_code, stderr)

        return stdout

    @contextlib.contextmanager
    def _tempfile_pdb(self, structure: Bio.PDB.Structure.Structure):
        """Write a .pdb structure to a temporary file inside the container"""
        with tempfile.NamedTemporaryFile(
            suffix=".pdb", dir=self.tmp_dir.name, mode="w"
        ) as f:
            path_outside = Path(f.name)
            path_inside = Path("/tmp") / path_outside.relative_to(self.tmp_dir.name)
            writer = Bio.PDB.PDBIO()
            writer.set_structure(structure)
            writer.save(f, preserve_atom_numbering=True)
            yield path_inside

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()


class DsspError(Exception):
    def __init__(self, exit_code: int, error_message: str):
        self.exit_code = exit_code
        self.error_message = error_message


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DSSP over a folder or a tar archive"
    )
    parser.add_argument("input_dir_or_tar", help="Directory or tar archive with .pdb files")
    parser.add_argument("output_dir", help="Directory for output .dssp files")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads")
    parser.add_argument("--verbose", "-v", action="count", default=0)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip .pdb file if the corresponding .dssp file exists",
    )
    args = parser.parse_args()
    return args


def run_dssp(input_dir_or_tar, output_dir, threads, verbose, skip_existing=False):
    """
    For each .pdb file compute DSSP features and save them as a .dssp file with the same name.

    Args:
        input_dir_or_tar: directory or tar archive with .pdb files
        output_dir: output directory (if input is a tar archive it will be extracted here)
        threads: number of joblib threads to parallelize the work
        verbose: verbosity of joblib
        skip_existing: skip .pdb file if the corresponding .dssp file exists

    """
    input_dir_or_tar = Path(input_dir_or_tar).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(exist_ok=True, parents=True)

    if input_dir_or_tar.is_file() and input_dir_or_tar.suffix in {".tar", ".tar.gz"}:
        pdb_files = extract_tar(input_dir_or_tar, output_dir)
        for p in pdb_files:
            if p.suffix == "":
                p.rename(p.with_suffix(".pdb"))
        input_dir = output_dir
    elif input_dir_or_tar.is_dir():
        input_dir = input_dir_or_tar
        pdb_files = list(input_dir.glob("**/*.pdb"))
    else:
        raise ValueError(f"Invalid input: {input_dir_or_tar}")

    logger.info(f"Running on {len(pdb_files)} pdb files")
    with warnings.catch_warnings():
        # Ignore PDB warnings about missing atom elements
        warnings.simplefilter("ignore", Bio.PDB.PDBExceptions.PDBConstructionWarning)

        # Create a single container where multiple dssp processes can run
        with DsspDocker(input_dir, output_dir, skip_existing) as dssp_container:

            # Use threads instead of processes for parallelization
            # since every thread simply has to start a process inside docker and wait.
            with Parallel(n_jobs=threads, verbose=verbose, prefer="threads") as pool:
                pool(delayed(dssp_container.run)(pdb_file) for pdb_file in pdb_files)


def main():
    args = parse_args()
    run_dssp(**vars(args))


"""
python -m graphqa.data.dssp pdb_files.tar.gz output_dir
"""
if __name__ == "__main__":
    main()
