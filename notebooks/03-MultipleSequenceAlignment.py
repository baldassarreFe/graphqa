# ---
# jupyter:
#   jupytext:
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
# # Multiple-sequence alignment
# Install [Jackhmmer](http://hmmer.org/documentation.html) with conda:
# ```bash
# conda install -c bioconda hmmer
# ```
#
# Get sequences database [UniRef50](ftp://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/) (7.3GB download, 15GB uncompressed):
# ```bash
# wget -q 'ftp://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz' 
# gzip --decompress 'uniref50.fasta.gz'
# ```
#
# Example run on one sequence:
#
# ```bash
# cat << EOF > '/tmp/T0780.fasta'
# >T0780 SP18142A - SP_1560, Streptococcus pneumoniae TIGR4, 259 residues
# MKKNSLYIISSLFFACVLFVYATATNFQNSTSARQVKTETYTNTVTNVPIDIRYNSDKYF
# ISGFASEVSVVLTGANRLSLASEMQESTRKFKVTADLTDAGVGTIEVPLSIEDLPNGLTA
# VATPQKITVKIGKKAQKDKVKIVPEIDPSQIDSRVQIENVMVSDKEVSITSDQETLDRID
# KIIAVLPTSERITGNYSGSVPLQAIDRNGVVLPAVITPFDTIMKVTTKPVAPSSSTSNSS
# TSSSSETSSSTKATSSKTN
# EOF
#
# jackhmmer -o '/tmp/jackhmmer.out' -A '/tmp/alignments.sto' -N 3 -E .001 --cpu 8 '/tmp/T0780.fasta' 'uniref50.fasta'
# ```

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

pd.set_option("display.max_columns", 25)

# %%
alignments = Bio.AlignIO.parse("/tmp/alignments.sto", "stockholm")
for alignment in alignments:
    if alignment[0].id == 'T0780':
        break

original_sequence = [aa for aa in alignment[0] if aa != '-']
print(alignment)


# %% [markdown]
# ## Frequency count for each residue
#
# BioPython implementation

# %%
def msa_counts_bio(alignment):
    original_sequence = [aa for aa in alignment[0] if aa != '-']
    summary_align = Bio.Align.AlignInfo.SummaryInfo(alignment)
    pssm = summary_align.pos_specific_score_matrix(alignment[0], chars_to_ignore=["-"])
    msa_counts = pd.DataFrame(
        [counts for aa, counts in pssm.pssm if aa != '-'], 
        dtype=np.int, 
        index=original_sequence,
    )
    return msa_counts

# %timeit msa_counts_bio(alignment)

msa_counts = msa_counts_bio(alignment)
msa_counts

# %% [markdown]
# Faster

# %%
bins = np.arange(22)
msa_1_mapping = {**aa_1_mapping, 'X': 21, '-': 22}
msa_1_mapping_inv = aa_1_mapping_inv + ['X']

def msa_counts_np(alignment):
    original_sequence = [aa for aa in alignment[0] if aa != "-"]
    seq_length = len(original_sequence)
    msa_counts = np.empty((seq_length, 21), dtype=np.int)
    
    idx_in_seq = 0
    for idx_in_msa, aa in enumerate(alignment[0]):
        if aa == '-':
            continue
        
        msa_at_idx = [msa_1_mapping[seq[idx_in_msa]] for seq in alignment]
        counts = np.histogram(msa_at_idx, bins=bins)[0]
        
        msa_counts[idx_in_seq] = counts
        idx_in_seq += 1
    
    msa_counts = pd.DataFrame(msa_counts, index=original_sequence, columns=msa_1_mapping_inv)
    return msa_counts

# %timeit msa_counts_np(alignment)

msa_counts = msa_counts_np(alignment)
msa_counts.sort_index(axis='columns')

# %% [markdown]
# ## Partial entropy
# $a$ = amino acid
#
# $i$ = position in the sequence
#
# $I_{i, a} = - p_{i, a} \log\frac{p_{i, a}}{\sum_j p_{j, a}}$
#
# Below is not partial entropy, it's just normalized frequencies

# %%
msa_counts = msa_counts.values
freq = msa_counts.astype(np.float) / msa_counts.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(1, 1, figsize=(26, 4), tight_layout=True)
ax.pcolormesh(freq.T, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks(np.arange(len(original_sequence))+.5)
ax.set_xticklabels(original_sequence)
ax.set_yticks(np.arange(len(aa_1_mapping_inv) + 1)+.5)
ax.set_yticklabels(aa_1_mapping_inv + ['X'])

fig.set_facecolor('white')
display(fig)
plt.close(fig)


# %% [markdown]
# ## Process all data

# %% language="bash"
# for ed in 9 10 11 12 13; do
#   jackhmmer -N 3 -E .001 --incE 0.001 --cpu 4 \
#     -o "CASP${ed}/jackhmmer.out" \
#     -A "CASP${ed}/alignments.sto" \
#     "CASP${ed}/sequences.fasta" 'uniref50.fasta' &
# done
# wait
#
# du -shc CASP*/alignments.sto

# %%
def compute_and_save(casp_ed):
    path = f'CASP{casp_ed}/alignments.sto'
    all_msa_counts = {}
    for alignment in Bio.AlignIO.parse(path, format="stockholm"):
        target_id = alignment[0].id
        msa_counts = msa_counts_np(alignment)
        all_msa_counts[target_id] = msa_counts
    pd.to_pickle(all_msa_counts, f'CASP{casp_ed}/alignments.pkl')

with Parallel(n_jobs=5) as pool:
    pool(delayed(compute_and_save)(casp_ed) for casp_ed in [9, 10, 11, 12, 13])
    
# ! rm CASP*/alignments.sto
# ! du -shc CASP*/alignments.pkl
