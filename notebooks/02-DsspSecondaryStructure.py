# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Secondary structure
#
# Protein [secondary structure](https://en.wikipedia.org/wiki/Protein_secondary_structure) is the three dimensional form of _local_ segments of proteins.
#
# [DSSP](https://swift.cmbi.umcn.nl/gv/dssp/index.html) determines properties of the secondary structure given the three dimensional coordinates of a protein. It does not _predict_ secondary structure, just _describes_ it.

# %%
# %matplotlib agg
import io
import os
import re
import time
import json
import hashlib
import tarfile
import requests
import tempfile
import warnings
import functools
import contextlib
import subprocess
from pathlib import Path

import bs4
import docker
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation
import natsort as ns
import tqdm.notebook as tqdm

import Bio.PDB
import Bio.SeqIO
import Bio.Align.AlignInfo
import Bio.AlignIO

from loguru import logger
from joblib import Parallel, delayed
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display, Markdown, HTML, Video

from graphqa.data.aminoacids import *

# %% [markdown]
# ## [States](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6090794/)
# Protein secondary structures are traditionally characterized as 3 general states: helix (H), strand (E), and coil (C).
#
# The DSSP program uses a finer characterization of the secondary structures by extending the 3 states into 8 states:
# - Helix:
#   - `G` = 3-turn helix (310 helix). Min length 3 residues.
#   - `H` = 4-turn helix (α helix). Minimum length 4 residues.
#   - `I` = 5-turn helix (π helix). Minimum length 5 residues.
# - Strand:
#   - `T` = hydrogen bonded turn (3, 4 or 5 turn)
#   - `E` = extended strand in parallel and/or anti-parallel β-sheet conformation. Min length 2 residues.
#   - `B` = residue in isolated β-bridge (single pair β-sheet hydrogen bond formation)
#   - `S` = bend (the only non-hydrogen-bond based assignment)
# - Coil:
#   - `-` = coil (residues which are not in any of the above conformations).
#
# ## [Dihedral angles](https://en.wikipedia.org/wiki/Dihedral_angle)
# <img src="https://upload.wikimedia.org/wikipedia/commons/9/97/Protein_backbone_PhiPsiOmega_drawing.svg" style="background-color:white;width:10%;"/>

# %% [markdown]
# Get [DSSP](https://swift.cmbi.umcn.nl/gv/dssp/) and build a docker image
# ```bash
# git clone https://github.com/cmbi/dssp ~/dssp
# pushd ~/dssp
# git checkout 697deab74011bfbd55891e9b8d5d47b8e4ef0e38
# docker build -t dssp .
# popd
# ```
#
# Important:
# - `mkdssp` only reads from file, not from stdin
# - `mkdssp` skips lines shorter than 80 chars, some pdb files contains shorter lines but rewriting them with BioPython fixes the problem
#
# Start a container with the current folder with the pdb files mounted as `/data`:

# %% {"language": "bash"}
# (docker top dssp && docker stop dssp) 2>&1 > /dev/null
# docker run --rm --tty --detach \
#   --name dssp \
#   --mount "type=bind,source=$(realpath ../data),target=/data" \
#   'dssp'
# docker ps --filter "name=dssp"

# %%
docker_client = docker.from_env()
dssp_container = docker_client.containers.get("dssp")
df_decoys = pd.read_csv("decoys.csv")

# %% [markdown]
# `mkdssp` can be run inside the container, the output is buch of text that BioPython can parse for us (but only from a file on disk):

# %%
parser = Bio.PDB.PDBParser(QUIET=True)
structure = parser.get_structure(
    "T0759/3D-Jigsaw-V5_1_TS1", "CASP11/decoys/T0759/3D-Jigsaw-V5_1_TS1.pdb"
)

exit_code, (stdout, stderr) = dssp_container.exec_run(
    "/app/mkdssp /data/CASP11/decoys/T0759/3D-Jigsaw-V5_1_TS1.pdb", demux=True
)

# Raw output
print(
    *stdout.decode().splitlines()[:2],
    "...",
    *stdout.decode().splitlines()[27:35],
    "...",
    sep="\n",
)

# Parsed output
with tempfile.NamedTemporaryFile() as f:
    f.write(stdout)
    f.flush
    dssp = Bio.PDB.DSSP(structure[0], in_file=f.name, file_type="DSSP")

pd.DataFrame.from_dict(
    dssp.property_dict,
    orient="index",
    columns=(
        "dssp index",
        "amino acid",
        "secondary structure",
        "relative ASA",
        "phi",
        "psi",
        "NH_O_1_relidx",
        "NH_O_1_energy",
        "O_NH_1_relidx",
        "O_NH_1_energy",
        "NH_O_2_relidx",
        "NH_O_2_energy",
        "O_NH_2_relidx",
        "O_NH_2_energy",
    ),
).rename_axis(index="(chain_id, res_id)")

# %% [markdown]
# Two examples of problematic pdb files

# %%
for decoy_path in [
    "CASP9/decoys/T0515/FFAS03ss_TS4.pdb",
    "CASP9/decoys/T0515/panther_TS1.pdb",
]:
    print(decoy_path)
    print("  Original:")
    with open(decoy_path) as f:
        print(*(f"    |{l[:-1]}|" for l in f.readlines()[4:8]), sep="\n")

    structure = parser.get_structure("tmp", decoy_path)
    writer = Bio.PDB.PDBIO()
    writer.set_structure(structure)
    writer.save("/tmp/tmp.pdb", preserve_atom_numbering=True)

    print("  Rewritten:")
    with open("/tmp/tmp.pdb") as f:
        print(*(f"    |{l[:-1]}|" for l in f.readlines()[:4]), sep="\n")
    print()


# %%
@logger.catch(reraise=False)
def run_dssp_in_docker(decoy_path: str, output_path: str):
    docker_client = docker.from_env()
    dssp_container = docker_client.containers.get("dssp")
    
    exit_code, (stdout, stderr) = dssp_container.exec_run(
        cmd=["/app/mkdssp", "/data/" + decoy_path], demux=True
    )

    if exit_code != 0:
        # Try a reformatted version of the decoy
        parser = Bio.PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("tmp", decoy_path)
        
        new_decoy_path = f"{decoy_path}_tmp.pdb"
        writer = Bio.PDB.PDBIO()
        writer.set_structure(structure)
        writer.save(new_decoy_path, preserve_atom_numbering=True)
        exit_code, (stdout, stderr) = dssp_container.exec_run(
            cmd=["/app/mkdssp", "/data/" + new_decoy_path], demux=True
        )
        Path(new_decoy_path).unlink()

        if exit_code != 0:
            logger.error(
                "DSSP error {}: {}",
                decoy_path,
                stderr.decode().strip() if stderr is not None else "no stderr",
            )
            return

    with open(output_path, "wb") as f:
        f.write(stdout)

with warnings.catch_warnings():
    # Ignore PDB warnings about missing atom elements
    warnings.simplefilter("ignore", Bio.PDB.PDBExceptions.PDBConstructionWarning)

    with Parallel(n_jobs=30, verbose=1, prefer="threads") as pool:
        missing_decoys = [
            dict(
                decoy_path=f"CASP{decoy.casp_ed}/decoys/{decoy.target_id}/{decoy.decoy_id}.pdb",
                output_path=f"CASP{decoy.casp_ed}/decoys/{decoy.target_id}/{decoy.decoy_id}.dssp",
            )
            for decoy in df_decoys.itertuples()
            if not Path(
                f"CASP{decoy.casp_ed}/decoys/{decoy.target_id}/{decoy.decoy_id}.dssp"
            ).is_file()
        ]
        logger.info(f"Running on {len(missing_decoys)} .pdb files")
        pool(delayed(run_dssp_in_docker)(**decoy_dict) for decoy_dict in missing_decoys)

# %% [markdown]
# Check how many `.pdb` and `.dssp` files we have to see where DSSP failed

# %%
pdb = set(p.with_suffix("") for p in Path().glob("CASP*/decoys/*/*.pdb"))
dssp = set(p.with_suffix("") for p in Path().glob("CASP*/decoys/*/*.dssp"))
failed = pdb - dssp
if failed:
    logger.warning(
        f"DSSP failed on {len(failed)}/{len(pdb)} decoys ({len(failed)/len(pdb):.2%})"
    )

# %% [markdown]
# Stop the container

# %%
dssp_container.stop()
