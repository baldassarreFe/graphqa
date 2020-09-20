# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # LDDT
#
# We use a docker container with [OpenStructure (v2.1.0)](https://git.scicore.unibas.ch/schwede/openstructure/-/tree/master/docker) for computing LDDT scores.
#
# Inside the container we run a custom python script adapted from one of the examples on the website.
#
# LDDT scores, both global and local will be saved as a `CASP*/decoys/<target_id>.lddt.npz` file containing:
# - `decoys`: 1D array of decoy names
# - `global_lddt`: 1D array of global scores
# - `local_lddt`: 2D array of local scores of shape `num_decoys x seq_length`

# %%
from pathlib import Path

import docker
import numpy as np
import pandas as pd

from loguru import logger
from joblib import Parallel, delayed

from graphqa.data.aminoacids import *

docker_client = docker.from_env()

# %% [markdown]
# Pull the [OpenStructure](https://www.openstructure.org/docs/2.0/install/) docker image and start a container with the LDDT python script mounted inside:

# %% {"language": "bash"}
# docker pull -q 'registry.scicore.unibas.ch/schwede/openstructure:2.1.0'
# docker stop lddt 2> /dev/null
# docker run --rm --tty --detach \
#   --name 'lddt' \
#   --entrypoint 'bash' \
#   --mount "type=bind,source=$(realpath ../src/graphqa/data/lddt_docker.py),target=/lddt.py" \
#   --mount "type=bind,source=${PWD},target=/input" \
#   --mount "type=bind,source=${PWD},target=/output" \
#   'registry.scicore.unibas.ch/schwede/openstructure:2.0.0'
# docker ps --filter "name=lddt"

# %%
lddt_container = docker_client.containers.get("lddt")
df_natives = pd.read_csv("natives_casp.csv")
target_lengths = pd.read_csv("sequences.csv").set_index("target_id").length.to_dict()


# %%
def run_lddt_in_docker(seq_len, native_path, decoys_dir, output_path):
    exit_code, (stdout, stderr) = lddt_container.exec_run(
        cmd=["/lddt.py", str(seq_len), native_path, decoys_dir, output_path], demux=True
    )

    if exit_code != 0:
        logger.error(f"LDDT error {native_path}: {stderr.decode()}")


with Parallel(n_jobs=10, prefer="threads") as pool:
    missing_targets = [
        dict(
            seq_len=target_lengths[target.target_id],
            native_path=f"CASP{target.casp_ed}/native/{target.target_id}.pdb",
            decoys_dir=f"CASP{target.casp_ed}/decoys/{target.target_id}",
            output_path=f"CASP{target.casp_ed}/decoys/{target.target_id}.lddt.npz",
        )
        for target in df_natives.itertuples()
        if not Path(
            f"CASP{target.casp_ed}/decoys/{target.target_id}.lddt.npz"
        ).is_file()
    ]
    logger.info(f"Launching {len(missing_targets)} jobs")
    pool(delayed(run_lddt_in_docker)(target_dict) for target_dict in missing_targets)

pdb = set(p.with_suffix("").name for p in Path().glob("CASP*/native/*.pdb"))
lddt = set(p.with_suffix("").name for p in Path().glob("CASP*/decoys/*.lddt.npz"))
for fail in pdb - lddt:
    logger.warning(f"LDDT failed on: {fail}")

# %% {"language": "bash"}
# docker stop lddt
