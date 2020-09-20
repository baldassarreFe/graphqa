# ---
# jupyter:
#   jupytext:
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
# # CAD Score
#
# CAD scores, both global and local will be saved as a `CASP*/decoys/<target_id>.cad.npz` file containing:
# - `decoys`: 1D array of decoy names
# - `global_cad`: 1D array of global scores
# - `local_cad`: 2D array of local scores of shape `num_decoys x seq_length`
#
# We use the "AS" version of CAD scores from [Voronota](https://kliment-olechnovic.github.io/voronota/):
# ```bash
# wget -q -O - 'https://github.com/kliment-olechnovic/voronota/releases/download/v1.21.2744/voronota_1.21.2744.tar.gz' | tar xz
# ```

# %%
from pathlib import Path

import numpy as np
import pandas as pd

from loguru import logger
from joblib import Parallel, delayed

from graphqa.data.aminoacids import *
import graphqa.data.cadscore as cadscore, CadScoreError

# %% [raw]
# ! rm CASP*/decoys/*.cad.npz

# %%
df_natives = pd.read_csv("natives_casp.csv")
target_lengths = pd.read_csv("sequences.csv").set_index("target_id")["length"].to_dict()


# %%
@logger.catch(reraise=False)
def compute_and_save_cad(
    native_path: str, decoys_dir: str, output_path: str, sequence_length: int
):
    try:
        run_cadscore(
            native_path, decoys_dir, sequence_length,
            voronota="voronota_1.21.2744/voronota-cadscore"
        )
    except CadScoreError as e:
        logger.warning(e)


with Parallel(n_jobs=20, verbose=1, prefer="threads") as pool:
    missing_targets = [
        dict(
            native_path=f"CASP{target.casp_ed}/native/{target.target_id}.pdb",
            decoys_dir=f"CASP{target.casp_ed}/decoys/{target.target_id}",
            output_path=f"CASP{target.casp_ed}/decoys/{target.target_id}.cad.npz",
            sequence_length=target_lengths[target.target_id],
        )
        for target in df_natives.itertuples()
        if not Path(f"CASP{target.casp_ed}/decoys/{target.target_id}.cad.npz").is_file()
    ]
    pool(
        delayed(compute_and_save_cad)(**target_dict) for target_dict in missing_targets
    )

# %%
pdb = set(p.with_suffix("").name for p in Path().glob("CASP*/native/*.pdb"))
cad = set(
    p.with_suffix("").with_suffix("").name
    for p in Path().glob("CASP*/decoys/*.cad.npz")
)
failed = pdb - cad
if failed:
    logger.warning(f"CAD score failed on {len(failed)} targets")
    if len(failed) < 20:
        for f in failed:
            logger.warning(f"CAD score failed on: {f}")

for p in Path().glob("CASP*/decoys/*.cad.npz"):
    count = len(np.load(p)["decoys"])
    if count < 20:
        logger.warning(f"{p} contains {count} decoys")
