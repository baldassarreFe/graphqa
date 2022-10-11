import tarfile
from pathlib import Path
from typing import Union, Callable, Optional, List

from loguru import logger


def sanitize_tarinfo(tarinfo: tarfile.TarInfo):
    path = Path(tarinfo.name)

    # Ban absolute paths
    if path.is_absolute():
        return False

    # Ban paths containing .. that would go outside
    try:
        base = Path("fake_path")
        base.joinpath(path).relative_to(base)
    except ValueError as e:
        logger.warning(f"Unsafe path {path}")
        return False

    # Ban links that would point somewhere outside
    if tarinfo.islnk() or tarinfo.issym():
        try:
            base = Path("fake_path")
            link = path.parent / tarinfo.linkname
            base.joinpath(link).relative_to(base)
        except ValueError as e:
            logger.warning(f"Unsafe path {path}")
            return False

    return True


def extract_tar(
    tar_path: Union[str, Path],
    output_path: Union[str, Path],
    accept: Optional[Callable[[tarfile.TarInfo], bool]] = None,
) -> List[Path]:
    tar_path = Path(tar_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()

    with tarfile.open(tar_path, mode="r:*") as archive:
        members = archive.getmembers()
        members = filter(sanitize_tarinfo, members)
        if accept is not None:
            members = filter(accept, members)
        members = list(members)
        
        import os
        
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(archive, output_path, members)
    return [output_path / ti.name for ti in members]
