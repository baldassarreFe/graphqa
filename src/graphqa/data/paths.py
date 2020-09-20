from pathlib import Path
from typing import Union, NamedTuple


class GraphQaPaths(NamedTuple):
    name: str
    root: Path
    sequences: Path
    alignments: Path
    decoys: Path
    native: Path
    processed: Path


def dataset_paths(dataset_dir: Union[str, Path]) -> GraphQaPaths:
    """
    Args:
        dataset_dir: root folder of a protein dataset

    Returns:
        a namedtuple of pathlib.Path entries
    """
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    return GraphQaPaths(
        name=dataset_dir.name,
        root=dataset_dir,
        sequences=dataset_dir / "sequences.fasta",
        alignments=dataset_dir / "alignments.pkl",
        decoys=dataset_dir / "decoys",
        native=dataset_dir / "native",
        processed=dataset_dir / "processed",
    )
