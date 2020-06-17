from typing import Union, Tuple
from pathlib import Path

import Bio.PDB
import Bio.PDB.Residue
import numpy as np
import pandas as pd
from loguru import logger

# (None, 0) means missing residue or missing DSSP features
ss_dssp_mapping = {
    None: 0,
    "G": 1,
    "H": 2,
    "I": 3,
    "T": 4,
    "E": 5,
    "B": 6,
    "S": 7,
    "-": 8,
}


def parse_pdb(path: Union[str, Path], id=""):
    path = Path(path).expanduser().resolve()
    parser = Bio.PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(id=id, file=path.as_posix())
    structure.get_models()
    return structure


def parse_dssp(path: Union[str, Path], structure):
    path = Path(path).expanduser().resolve()
    Bio.PDB.DSSP(structure[0], path, file_type="DSSP")
    return structure


def ca_coord_and_orientation(
    residue: Bio.PDB.Residue.Residue,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coordinates of a residue's carbon alpha and direction CA->CB.
    If the residue is GLY, the atom CB is "virtual".
    """

    ca = residue["CA"].get_vector()

    try:
        cb = residue["CB"].get_vector()
        orient = cb - ca
    except KeyError:
        if residue.resname != "GLY":
            logger.warning(f"{residue.resname} does not have a beta carbon")
        # Estimate orientation based on C and N
        n = residue["N"].get_vector() - ca
        c = residue["C"].get_vector() - ca
        # find rotation matrix that rotates n -120 degrees along the ca-c vector
        rot = Bio.PDB.rotaxis(-np.pi * 120 / 180, c)
        # apply rotation to ca - n vector
        orient = n.left_multiply(rot)

    ca = ca.get_array()
    orient = orient.get_array()

    orient_norm = np.linalg.norm(orient)
    if orient_norm != 0:
        orient /= orient_norm

    return ca, orient


def load_decoy_feats(pdb: Path, dssp: Path, seq_length: int):
    # Load structure and DSSP
    structure = parse_pdb(pdb)
    structure = parse_dssp(dssp, structure)

    # Tertiary structure features
    coords_ca = np.full((seq_length, 3), dtype=np.float32, fill_value=np.nan)
    orient_res = np.full((seq_length, 3), dtype=np.float32, fill_value=np.nan)

    # DSSP features
    secondary_structure = np.full(
        (seq_length,), dtype=np.uint8, fill_value=ss_dssp_mapping[None]
    )
    phi = np.full((seq_length,), dtype=np.float32, fill_value=np.nan)
    psi = np.full((seq_length,), dtype=np.float32, fill_value=np.nan)
    surface_acc = np.full((seq_length,), dtype=np.float32, fill_value=np.nan)

    # Assume there is only one model per file and only one chain per model
    model = next(structure.get_models())
    chain = next(model.get_chains())
    for residue in chain:
        _, idx, _ = residue.id
        idx = idx - 1

        # Tertiary structure features
        ca, orient = ca_coord_and_orientation(residue)
        coords_ca[idx] = ca
        orient_res[idx] = orient

        # Some residues don't have DSSP features
        if len(residue.xtra) > 0:
            secondary_structure[idx] = ss_dssp_mapping[residue.xtra["SS_DSSP"]]
            phi[idx] = residue.xtra["PHI_DSSP"]
            psi[idx] = residue.xtra["PSI_DSSP"]
            surface_acc[idx] = residue.xtra["EXP_DSSP_RASA"]

    df_decoy = pd.DataFrame(
        {
            ("coords_ca", "x"): coords_ca[:, 0],
            ("coords_ca", "y"): coords_ca[:, 1],
            ("coords_ca", "z"): coords_ca[:, 2],
            ("orient_res", "x"): orient_res[:, 0],
            ("orient_res", "y"): orient_res[:, 1],
            ("orient_res", "z"): orient_res[:, 2],
            ("dssp", "ss"): secondary_structure,
            ("dssp", "surface_acc"): surface_acc,
            ("dssp", "phi"): phi,
            ("dssp", "psi"): psi,
        },
        index=pd.RangeIndex(seq_length, name="residue_idx"),
    )
    return df_decoy
