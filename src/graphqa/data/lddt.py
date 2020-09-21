"""
Compute LDDT scores between a native structure and a folder of decoys using OpenStructure.

First pull a docker image of OpenStructure:
```bash
docker pull 'registry.scicore.unibas.ch/schwede/openstructure:2.1.0'
```

Run from the command line as:
```bash
python -m graphqa.data.lddt -vv \
  CASP9/native/T0522.pdb \
  CASP9/decoys/T0522/ \
  CASP9/decoys/T0522.lddt.npz \
  134
```
"""
import os
import sys
import argparse
from pathlib import Path
from typing import Union

import docker
from loguru import logger

logger.disable("graphqa.data.lddt")


class LddtDocker(object):
    def __init__(
        self,
        native_dir: Union[str, Path],
        decoy_dir: Union[str, Path],
        output_dir: Union[str, Path],
        skip_existing=False,
    ):
        self.native_dir = Path(native_dir).expanduser().resolve()
        self.decoy_dir = Path(decoy_dir).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.skip_existing = skip_existing

        self.client = docker.from_env()
        self.container = None
        self.image = self.client.images.get(
            "registry.scicore.unibas.ch/schwede/openstructure:2.1.0"
        )

    def start(self):
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        mounts = [
            docker.types.Mount(
                source=Path(__file__).parent.joinpath("lddt_docker.py").as_posix(),
                target="/lddt_docker.py",
                type="bind",
                read_only=True,
            ),
            docker.types.Mount(
                source=self.native_dir.as_posix(),
                target="/native",
                type="bind",
                read_only=True,
            ),
            docker.types.Mount(
                source=self.decoy_dir.as_posix(),
                target="/decoy",
                type="bind",
                read_only=True,
            ),
            docker.types.Mount(
                source=self.output_dir.as_posix(),
                target="/output",
                type="bind",
                read_only=False,
            ),
        ]

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

    def run(self, native_pdb, decoys_dir, output_npz, sequence_length: int):
        if not self.container:
            raise RuntimeError(
                "LDDT container is not running, call start() first "
                "or use this instance as context manager."
            )

        native_pdb = Path(native_pdb).expanduser()
        if native_pdb.is_absolute():
            native_pdb = native_pdb.relative_to(self.native_dir)

        decoys_dir = Path(decoys_dir).expanduser()
        if decoys_dir.is_absolute():
            decoys_dir = decoys_dir.relative_to(self.decoy_dir)

        output_npz = Path(output_npz).expanduser()
        if output_npz.is_absolute():
            output_npz = output_npz.relative_to(self.output_dir)

        exit_code, (stdout, stderr) = self.container.exec_run(
            cmd=[
                "/lddt_docker.py",
                str(sequence_length),
                native_pdb.as_posix(),
                decoys_dir.as_posix(),
                output_npz.as_posix(),
            ],
            demux=True,
        )

        if exit_code != 0:
            stdout = stdout.decode().strip() if stdout is not None else "<no stdout>"
            stderr = stderr.decode().strip() if stderr is not None else "<no stderr>"
            raise LddtError(exit_code, f"{stdout}\n{stderr}")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()


class LddtError(Exception):
    def __init__(self, exit_code: int, error_message: str):
        self.exit_code = exit_code
        self.error_message = error_message


class LogFilter(object):
    _mapping = {0: "WARNING", 1: "INFO", 2: "DEBUG", 3: "TRACE"}

    def __init__(self, verbose):
        level = LogFilter._mapping.get(verbose, "TRACE")
        self.level_no = logger.level(level).no

    def __call__(self, record):
        return record["level"].no >= self.level_no


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LDDT between a native structure and a folder of decoys"
    )
    parser.add_argument("native_pdb")
    parser.add_argument("decoys_dir")
    parser.add_argument("output_npz", help="Output path (numpy archive format)")
    parser.add_argument("sequence_length", type=int)
    parser.add_argument("--verbose", "-v", action="count", default=0)
    args = parser.parse_args()

    return args


def run_lddt(native_pdb, decoys_dir, output_npz, sequence_length: int):
    native_pdb = Path(native_pdb).expanduser().resolve()
    decoys_dir = Path(decoys_dir).expanduser().resolve()
    output_npz = Path(output_npz).expanduser().resolve()

    with LddtDocker(
        native_pdb.parent, decoys_dir.parent, output_npz.parent
    ) as lddt_container:
        lddt_container.run(
            native_pdb.name, decoys_dir.name, output_npz.name, sequence_length
        )


def main():
    args = vars(parse_args())

    logger.enable("graphqa.data.lddt")
    logger.remove()
    logger.add(sys.stderr, filter=LogFilter(args.pop("verbose")), level=0)

    run_lddt(**args)


if __name__ == "__main__":
    main()
