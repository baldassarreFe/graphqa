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
# # TM_Score, GDT_TS, GDT_HA
#
# TM_Score, GDT_TS, GDT_HA scores will be saved as a `CASP*/decoys/<target_id>.tmscore.tsv` file, in tabular format, one row per decoy.
#
# Download and compile the TMscore executable from [Zhang lab](https://zhanglab.ccmb.med.umich.edu/TM-score/):
# ```bash
# wget -q 'https://zhanglab.ccmb.med.umich.edu/TM-score/TMscore.cpp'
# g++ -static -O3 -ffast-math -lm -o TMscore TMscore.cpp
# rm TMscore.cpp
# ```

# %%
from pathlib import Path

import numpy as np
import pandas as pd

from loguru import logger
from joblib import Parallel, delayed

from graphqa.data.aminoacids import *
from graphqa.data.tmscore import run_tmscore, TmScoreError


# %% [markdown]
# Run in parallel on all targets of all CASP editions

# %%
@logger.catch(reraise=False)
def compute_and_save_tm(native_path: str, decoys_dir: str, output_path: str):
    try:
        run_tmscore(native_path, decoys_dir, output_path, tmscore="./TMscore")
    except TmScoreError as e:
        logger.warning(e)


df_natives = pd.read_csv("natives_casp.csv")
with Parallel(n_jobs=10, verbose=1, prefer="threads") as pool:
    missing_targets = [
        dict(
            native_pdb=f"CASP{target.casp_ed}/native/{target.target_id}.pdb",
            decoys_dir=f"CASP{target.casp_ed}/decoys/{target.target_id}",
            output_npz=f"CASP{target.casp_ed}/decoys/{target.target_id}.tmscore.npz",
        )
        for target in df_natives.itertuples()
        if not Path(
            f"CASP{target.casp_ed}/decoys/{target.target_id}.tmscore.npz"
        ).is_file()
    ]
    pool(delayed(compute_and_save_tm)(**target_dict) for target_dict in missing_targets)

# %% [markdown]
# Check how many targets failed

# %%
pdb = set(p.with_suffix("").name for p in Path().glob("CASP*/native/*.pdb"))
tmscore = set(
    p.with_suffix("").with_suffix("").name
    for p in Path().glob("CASP*/decoys/*.tmscore.npz")
)
failed = pdb - tmscore

if len(failed) > 0:
    logger.warning(f"TMscore failed on {len(failed)} targets")
    if len(failed) < 20:
        for f in failed:
            logger.warning(f"TMscore failed on: {f}")

for p in Path().glob("CASP*/decoys/*.tmscore.npz"):
    count = len(np.load(p)["decoys"])
    if count < 20:
        logger.warning(f"{p} contains {count} decoys")
