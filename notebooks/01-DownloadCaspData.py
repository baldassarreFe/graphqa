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
# # Download CASP data
#
# This notebook downloads all raw protein data from the CASP website.
#
# Each CASP edition is downloaded into `./CASP*` which is a symlink to `../data/CASP*`.

# %%
# %matplotlib agg
import io
import os
import re
import time
import json
import pickle
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
from graphqa.data.decoys import ca_coord_and_orientation

@functools.lru_cache(maxsize=128)
def requests_get(url):
    return requests.get(url)


# %% [markdown]
# Documentation of CASP download area for CASP 11

# %%
response = requests_get('https://predictioncenter.org/download_area/README')
readme = response.text.splitlines()
i = readme.index('CASP11')
print(*readme[i: i+12], sep='\n')

# %% [markdown] toc-hr-collapsed=true toc-nb-collapsed=true
# ## Primary structure
#
# Download all sequences from the CASP website and save them as `CASP*/sequences.fasta`.
#
# Metadata about all sequences is saved in [`sequences.csv`](./sequences.csv).

# %%
regex_name = re.compile(r'(T|H)\d\d\d\d')
regex_subunit = re.compile(r'\bsubunit\s+(\d+)')

def is_qa_target(target):
    return bool(regex_name.fullmatch(target.name))

def rename(target):
    original_name = target.name
    match = regex_name.fullmatch(target.name)
    if match.groups()[0] == 'H':
        subunit = regex_subunit.search(target.description).groups()[0]
        casp_id = target.name.replace('H', 'T') + 's' + subunit
        target.description = (
            target.description.replace(target.name, casp_id, 1)
            + f' (original target name {original_name})'
        )
        target.name = casp_id
        target.id = casp_id
    return target, original_name

df_sequences = []
for casp_ed in [9, 10, 11, 12, 13]:
    dest = Path(f'CASP{casp_ed}/sequences.fasta')
    dest.parent.mkdir(exist_ok=True, parents=True)
    
    response = requests_get(f'https://predictioncenter.org/download_area/'
                            f'CASP{casp_ed}/sequences/casp{casp_ed}.seq.txt')
    sequences = Bio.SeqIO.parse(io.StringIO(response.text), format='fasta')
    sequences = filter(is_qa_target, sequences)
    sequences = map(rename, sequences)
    
    with dest.open('w') as f:
        for seq, original_name in sequences:
            df_sequences.append({
                'casp_ed': casp_ed,
                'target_id': seq.id,
                'target_id_orig': original_name,
                'length': len(seq),
            })
            Bio.SeqIO.write(seq, f, format='fasta')

df_sequences = pd.DataFrame(df_sequences)
df_sequences.to_csv('sequences.csv', index=False)
df_sequences.groupby('casp_ed').size().rename_axis('Edition').to_frame('Targets')

# %% [markdown]
# In some cases the same protein (`target_id_orig`) corresponds to more targets (`target_id`):

# %%
df_sequences.query('target_id != target_id_orig')

# %% [markdown] toc-hr-collapsed=true toc-nb-collapsed=true
# ## Tertiary structure

# %% [markdown]
# ### Native structures

# %% [markdown]
# #### Downloaded from the CASP download area
#
# These native structures are downloaded from the CASP download area.
#
# Metadata about these structures is saved in [`natives_casp.csv`](./natives_casp.csv).
#
# - all names start with `T` and end with an optional subunit `s`
# - some of the published primary sequences don't have an official native structure (maybe canceled?)

# %%
parser = Bio.PDB.PDBParser(QUIET=True)
df_natives = []
# Match files like T0759s1.pdb but not T0759-D1.pdb
regex = re.compile(r'T\d\d\d\d(?:s\d)?.pdb')

native_urls = {
    9: 'https://predictioncenter.org/download_area/CASP9/targets/casp9.targ_unsplit.tgz',
    10: 'https://predictioncenter.org/download_area/CASP10/targets/casp10.targets_unsplitted.noT0695T0739.tgz',
    11: 'https://predictioncenter.org/download_area/CASP11/targets/casp11.targets_unsplitted.release11242014.tgz',
    12: 'https://predictioncenter.org/download_area/CASP12/targets/casp12.targets_T0.releaseDec022016.tgz',
    13: 'https://predictioncenter.org/download_area/CASP13/targets/casp13.targets.T.4public.tar.gz',
}

for casp_ed, url in native_urls.items():
    dest = Path(f'CASP{casp_ed}') / 'native'
    dest.mkdir(exist_ok=True, parents=True)
    # ! curl -s {url} | tar xz --directory {dest.as_posix()}
    
    if casp_ed == 13:
        try:
            import casp13_secret
            # ! echo '--user {casp13_secret.user}:{casp13_secret.pwd}' | curl -s --config - {casp13_secret.url} | tar xz --directory {dest}
        except ImportError:
            print("Could not get limited-access CASP13 files")

    for f in dest.iterdir():
        if not regex.fullmatch(f.name):
            f.unlink()
            continue
        target_id = f.with_suffix('').name
        structure = parser.get_structure(target_id, f)
        df_natives.append({
            'casp_ed': casp_ed,
            'target_id': target_id,
            'chains': len(list(structure.get_chains())),
            'residues': len(list(structure.get_residues())),
            'atoms': len(list(structure.get_atoms())),
        })
        
df_natives = pd.DataFrame(df_natives).sort_values(['casp_ed', 'target_id'])
df_natives.to_csv('natives_casp.csv', index=False)
df_natives.groupby('casp_ed').size().rename_axis('Edition').to_frame('Targets')

# %% [markdown]
# A full outer join between the sequences in `.fasta` format 
# and the sequences found in the native's `.pdb` file
# shows the some discrepancies.
#
# In the following:
# - **left:**  primary sequences published on the CASP website
# - **right:** native structures published on the CASP website

# %%
df_merge = pd.merge(
    df_sequences, 
    df_natives.drop(columns=['chains', 'atoms']), 
    on=['casp_ed', 'target_id'], how='outer', indicator=True
)

# Some targets are present in the .fasta files but absent from the native .pdb files
display(
    df_merge['_merge']
    .value_counts()
    .to_frame('Outer join counts')
)
display(df_merge.query("_merge!='both'"))

# Some targets have different lenghts in the .fasta and .pdb files
display(
    (df_merge['length'] == df_merge['residues'])
    .value_counts()
    .to_frame('Same length?')
)
display(df_merge.query('length != residues and _merge=="both"'))

del df_merge

# %% [markdown]
# #### Downloaded from the Protein Data Bank
#
# These native structures are downloaded from the Protein Data Bank using 
# [the mapping from CASP id to PDB code can](https://predictioncenter.org/casp11/targetlist.cgi?view_targets=all).
#
# Metadata about these structures is saved in [`natives_pdb.csv`](./natives_pdb.csv).
#
# - Some targets are listed using `H????` names and some others using `T????[s?]` names, we only download the latter
# - Some structures downloaded from PDB actually have more than one model in the same `.pdb` file (but CASP only cares about one, right?)

# %%
# Match names like T0759s1
regex_name = re.compile(r'T\d\d\d\d(?:s\d)?')
parser = Bio.PDB.PDBParser(QUIET=True)
logger.disable('__main__:extract_targets')
df_natives_pdb = []

def extract_targets(soup):
    for target_row in soup.select('tr.datarow'):
        target_cols = target_row.select('td')
        
        try:
            casp_id = target_cols[1].select_one('a').text.strip()
            if not regex_name.match(casp_id):
                continue
        except Exception:
            msg = re.sub('\s+', ' ', target_cols[1].text).strip()
            logger.exception('Could not parse target id: ' + msg)
            continue
            
        try:
            length = int(target_cols[3].text.strip())
        except Exception:
            msg = re.sub('\s+', ' ', target_cols[3].text).strip()
            logger.exception('Could not parse length: ' + msg)
            continue
                
        try:
            txt = target_cols[-1].text.lower()
            if 'no structure' in txt or 'canceled' in txt:
                continue            
            pdb_code = target_cols[-1].select_one('a').text.strip()
        except Exception:
            msg = msg = re.sub('\s+', ' ', target_cols[-1].text).strip()
            logger.warning('Could not parse PDB code: ' + msg)
            continue
                        
        yield {
            'target_id': casp_id,
            'length': length,
            'pdb_id': pdb_code,
        }
        

def download_native(target_dict, dest):
    if not dest.is_file():
        response = requests.get(f'https://files.rcsb.org/download/{target_dict["pdb_id"]}.pdb')
        with dest.open('w') as f:
            f.write(response.text)

for casp_ed in [9,10,11,12,13]:
    with logger.contextualize(casp=casp_ed):
        dest = Path(f'CASP{casp_ed}') / 'native_pdb'
        dest.mkdir(exist_ok=True, parents=True)

        response = requests_get(f'https://predictioncenter.org/casp{casp_ed}/targetlist.cgi?view_targets=all')
        soup = bs4.BeautifulSoup(response.content)

        for target_dict in extract_targets(soup):
            dest_path = dest / f'{target_dict["target_id"]}.pdb'
            download_native(target_dict, dest_path)
            structure = parser.get_structure(target_dict["target_id"], dest_path)
            df_natives_pdb.append({
                'casp_ed': casp_ed,
                'chains': len(list(structure.get_chains())),
                'residues': len(list(structure.get_residues())),
                'atoms': len(list(structure.get_atoms())),
                **target_dict
            })
        
df_natives_pdb = pd.DataFrame(df_natives_pdb)
df_natives_pdb.to_csv('natives_pdb.csv', index=False)
df_natives_pdb.groupby('casp_ed').size().rename_axis('Edition').to_frame('Targets')

# %% [markdown]
# Difference between:
# - **left:**  primary sequences published on the CASP website
# - **right:** native structures downloaded from PDB

# %%
df_merge = pd.merge(
    df_sequences, 
    df_natives_pdb.drop(columns=['chains', 'atoms','length']),
    suffixes=['_fasta', '_casp'],
    on=['casp_ed', 'target_id'],
    how='outer', indicator=True
)

display(
    df_merge['_merge']
    .value_counts()
    .to_frame('Outer join counts')
)
display(df_merge.query("_merge!='both'"))

display(
    (df_merge['length'] == df_merge['residues'])
    .value_counts()
    .to_frame('Same length?')
)
display(df_merge.query('length != residues'))

del df_merge


# %% [markdown]
# ### Server predictions
#
# These are the tertiary structures as predicted from the servers participating in CASP
# ([submission file format](https://predictioncenter.org/casp13/index.cgi?page=format#TS)).
#
# The submission happens in two stages, the same target might get different names in the two stages.
#
# Each server can submit up to 5 models for each target, as indicated by the field `MODEL`.

# %%
@contextlib.contextmanager
def read_archive(response):
    with io.BytesIO(response.content) as fileobj:
        with tarfile.open(fileobj=fileobj, mode='r') as archive:
            yield archive

df_decoys = {}
for target in df_natives.itertuples():
    dest_dir = Path(f'CASP{target.casp_ed}/decoys/{target.target_id}')    
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Download all decoys for this target, compute md5sum and save them to disk
    response = requests_get(
        f'https://predictioncenter.org/download_area/'
        f'CASP{target.casp_ed}/server_predictions/'
        f'{target.target_id}.3D.srv.tar.gz'
    )
    if response.status_code != 200:
        logger.warning(f'{response.url} {response.status_code}')
        continue
    with read_archive(response) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            if Path(member.name).is_absolute():
                logger.warning(f'Invalid path in tarfile: {response.url} {member.name}')
                continue

            decoy_id = Path(member.name).with_suffix('').name
            with archive.extractfile(member) as fileobj:
                content = fileobj.read()
            with dest_dir.joinpath(decoy_id).with_suffix('.pdb').open('wb') as out:
                out.write(content)
            df_decoys[hashlib.md5(content).digest()] = {
                'casp_ed': target.casp_ed,
                'target_id': target.target_id,
                'decoy_id': decoy_id,
                'stage1': False,
                'stage2': False,
            }

    if target.casp_ed == 9:
        continue

    # Download all stage1 and stage2 decoys for this target,
    # don't save to disk, just compute md5sum and update the dict
    for s in ['stage1', 'stage2']:
        response = requests_get(
            f'https://predictioncenter.org/download_area/'
            f'CASP{target.casp_ed}/server_predictions/'
            f'{target.target_id}.{s}.3D.srv.tar.gz'
        )
        if response.status_code != 200:
            logger.warning(f'{response.url} {response.status_code}')
            continue
        with read_archive(response) as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                if Path(member.name).is_absolute():
                    logger.warning(f'Invalid path in tarfile: {response.url} {member.name}')
                    continue
                with archive.extractfile(member) as fileobj:
                    content = fileobj.read()
                try:
                    df_decoys[hashlib.md5(content).digest()][s] = True
                except KeyError:
                    logger.error(f'Could not find stage {s[-1]} decoy {member.name} '
                                 f'among all decoys of CASP{target.casp_ed}/{target.target_id}')
    
df_decoys = pd.DataFrame(df_decoys.values())
df_decoys.to_csv('decoys.csv', index=False)

# %%
# ! du -shc CASP*/decoys

df_decoys = pd.read_csv('decoys.csv')
display(
    df_decoys.groupby('casp_ed')
    .agg({'target_id': 'nunique', 'decoy_id': 'size', 'stage1': 'sum', 'stage2': 'sum'})
    .rename(columns={'target_id': 'Unique targets', 'decoy_id': 'Total decoys', 
                     'stage1': 'Decoys in stage 1', 'stage2': 'Decoys in stage 2'})
    .rename_axis('Edition')
    .astype(int)
)
display(
    df_decoys.groupby(['casp_ed', 'target_id'])
        .agg({'stage1': 'sum', 'stage2': 'sum', 'decoy_id': 'size'})
        .rename(columns={'decoy_id': 'Total', 'stage1': 'Stage 1', 'stage2': 'Stage 2'})
        .rename_axis(['Edition', 'Target'])
        .astype(int)
)

# %% [markdown]
# ### Visualization
#
# Focus on `CASP11/T0759` and its decoy `T0759/3D-Jigsaw-V5_1_TS1` as an example.
#
# For `CASP11/T0759` we have the following decoys:

# %%
display(
    df_decoys.query('target_id == "T0759"')
    .agg({'stage1': 'sum', 'stage2': 'sum', 'decoy_id': 'size'})
    .rename({'decoy_id': 'Total', 'stage1': 'Stage 1', 'stage2': 'Stage 2'})
    .to_frame('T0759')
    .transpose()
)

# %% [markdown]
# Draw the coordinates of a residue's carbon alpha and direction `CA->CB`.
# If the residue is `GLY`, the atom `CB` is "virtual".
#
# <img src="https://vignette.wikia.nocookie.net/foldit/images/8/86/Backbone_overview.stickpolarh.png/revision/latest?cb=20180101214816" style="background-color:white;width:15%;"/>

# %% [markdown]
# Native structure:

# %%
parser = Bio.PDB.PDBParser(QUIET=True)
model = parser.get_structure('T0759', 'CASP11/native/T0759.pdb')[0]

fig = plt.figure(figsize=(12, 6), facecolor='white', tight_layout=True)
ax = fig.add_subplot(1, 1, 1, projection='3d')

for chain_idx, chain in enumerate(model):
    aa = [r.get_resname() for r in chain if Bio.PDB.is_aa(r)]
    ca, orient = zip(*(ca_coord_and_orientation(r) for r in chain if Bio.PDB.is_aa(r)))
    ca = np.stack(ca, axis=0)    
    ax.plot(*ca.T, color=plt.get_cmap('tab10')(chain_idx))
    for ca, aa, orient in zip(ca, aa, orient):
        ax.plot(*zip(ca.T, (ca+orient).T), linewidth=3, color=plt.get_cmap('tab20')(aa_3_mapping[aa]))

animate = lambda i: ax.view_init(30, i)
ani = matplotlib.animation.FuncAnimation(fig, animate, frames=10 + 20 * np.sin(np.linspace(0, np.pi, num=100)), interval=50)
ani.save('T0759.mp4')
display(Video('T0759.mp4', embed=False))
plt.close(fig)

# %% [markdown]
# Decoy structure:

# %%
regex_target = re.compile(r'TARGET\s+(T\d\d\d\d(:?s\d+)?)')
regex_model = re.compile(r'MODEL\s+(\d+)')

with open('CASP11/decoys/T0759/3D-Jigsaw-V5_1_TS1.pdb') as f:
    # Print the first lines of the pdb file
    print(*f.readlines()[:9], sep='')

    # The parser does not recognize TARGET and MODEL fields automatically
    f.seek(0)
    f.readline()
    target_id = regex_target.match(f.readline()).groups()[0]
    model_id = regex_model.match(f.readline()).groups()[0]

    # Parse the structure
    f.seek(0)
    structure = parser.get_structure(f'T0759/3D-Jigsaw-V5_1_TS1', f)


# Print parsed structure
for model in structure:
    print(f'Model {model.get_full_id()} ({len(model)} chains)')
    for chain in model:
        print(f'  Chain {chain.get_full_id()} ({len(chain)} residues)')
        chain = list(chain)
        for residue in chain[:5]:
            print(f'   {residue.get_id()[1]:>3} {residue.get_resname()}  {len(residue):>2} atoms')
        print('   ...')
        for residue in chain[-5:]:
            print(f'   {residue.get_id()[1]:>3} {residue.get_resname()}  {len(residue):>2} atoms')
            

# Show animation
fig = plt.figure(figsize=(12, 6), facecolor='white', tight_layout=True)
ax = fig.add_subplot(1, 1, 1, projection='3d')

model = structure[0]
for chain_idx, chain in enumerate(model):
    aa = [r.get_resname() for r in chain]
    ca, orient = zip(*(ca_coord_and_orientation(r) for r in chain))
    ca = np.stack(ca, axis=0)    
    ax.plot(*ca.T, c='gray')
    for ca, aa, orient in zip(ca, aa, orient):
        ax.plot(*zip(ca.T, (ca+orient).T), linewidth=3, color=plt.get_cmap('tab20')(aa_3_mapping[aa]))

animate = lambda i: ax.view_init(30, i)
ani = matplotlib.animation.FuncAnimation(fig, animate, frames=10 + 20 * np.sin(np.linspace(0, np.pi, num=100)), interval=50)
ani.save('T0759_3D-Jigsaw-V5_1_TS1.mp4')
display(Video('T0759_3D-Jigsaw-V5_1_TS1.mp4', embed=False))
plt.close(fig)

# %% [markdown] toc-hr-collapsed=true toc-nb-collapsed=true
# ## Official QA global scores (summary tables)
#
# Summary tables contain official QA scores computed by CASP by comparison with the native structure for all tertiary structure predictions from all participants.
# QA metrics are only computed for global scores.
#
# A unique model `ACCESSION CODE` is composed from the number of the target, prediction format category, prediction group number, and model index. 
# Example:
# ```
# Accession code  T0444TS005_2  has the following components:
#  T0044   target number
#  TS      Tertiary Structure (PFRMAT TS)
#  005     prediction group 5
#  2       model index 2 
# ```
#
# Summary tables are a bit different in each CASP edition.
#
# For each edition, the table is saved as `CASP*/QA_official/table.pkl.xz`

# %%
# Match files like T0759s1.txt but not T0759-D1.txt
regex = re.compile(r'T\d\d\d\d(?:s\d)?.txt')

# %% [markdown]
# ### CASP 9
# Single table with all targets and decoys

# %%
url = 'https://predictioncenter.org/download_area/CASP9/refinement_result_tables_assessor.txt'
response = requests_get(url)

df = pd.read_csv(io.BytesIO(response.content), sep='\t')
df.rename(columns={c: c.strip() for c in df.columns}, inplace=True)
df.rename(columns={'GDT-HA': 'GDT_HA', 'GDC-SC': 'GDT_SC'}, inplace=True)
df['Exclude(missing too many atoms)'] = (
    df['Exclude(missing too many atoms)']
    .str.strip()
    .map({'True': True, 'False': False})
)
df['Target'] = df['Target'].map('T{:04d}'.format)
df['Model'] = df['Model'].astype(str)
df.set_index(['Target', 'Group', 'Model'], inplace=True)
df.sort_index(inplace=True)

Path('CASP9/QA_official').mkdir(exist_ok=True, parents=True)
df.to_pickle('CASP9/QA_official/table.pkl.xz')

print('Unique targets:', len(df.index.unique(level='Target')))
print('Unique groups:', len(df.index.unique(level='Group')))
df

# %% [markdown]
# ### CASP 10
# Single .tar.gz file with separate .txt files inside

# %%
url = 'https://predictioncenter.org/download_area/CASP10/SUMMARY_TABLES/T0_all.tar.gz'
response = requests_get(url)
archive = tarfile.open(fileobj=io.BytesIO(response.content), mode='r')

dfs = {}
for member in archive.getmembers():
    if member.isfile() and regex.fullmatch(member.name):
        f = archive.extractfile(member)
        df = pd.read_csv(f, sep='\s+')               
        dfs[member.name] = df
df = pd.concat(dfs.values(), axis=0)

# Split accession codes
split = df['Model'].str.split('TS', expand=True)
df['Target'] = split[0]
split = split[1].str.split('_', expand=True, n=1)
df['Group'] = split[0].astype(int)
df['Model'] = split[1]
df.set_index(['Target', 'Group', 'Model'], inplace=True)
df.sort_index(inplace=True)

Path('CASP10/QA_official').mkdir(exist_ok=True, parents=True)
df.to_pickle('CASP10/QA_official/table.pkl.xz')

print('Unique targets:', len(df.index.unique(level='Target')))
print('Unique groups:', len(df.index.unique(level='Group')))
df

# %% [markdown]
# ### CASP 11
# Single .tar.gz file with separate .txt files inside, 
# but also has some additional targets that are not in the .tar.gz file

# %%
url = 'https://predictioncenter.org/download_area/CASP11/SUMMARY_TABLES/T0xxx_09.05.tar.gz'
response = requests_get(url)
archive = tarfile.open(fileobj=io.BytesIO(response.content), mode='r')

dfs = {}
for member in archive.getmembers():
    if member.isfile() and regex.fullmatch(member.name):
        f = archive.extractfile(member)
        df = pd.read_csv(f, sep='\s+')
        dfs[member.name] = df
        
# Additional results that in the .tar.gz file are missing or outdated
url = 'https://predictioncenter.org/download_area/CASP11/SUMMARY_TABLES/'
for f in ['T0774.txt', 'T0812.txt', 'T0837.txt', 
          'T0840.txt', 'T0841.txt', 'T0851.txt']:
    response = requests_get(url + f)
    df = pd.read_csv(io.BytesIO(response.content), sep='\s+')
    dfs[f] = df
    
df = pd.concat(dfs.values(), axis=0)

# Split accession codes
split = df['Model'].str.split('TS', expand=True)
df['Target'] = split[0]
split = split[1].str.split('_', expand=True, n=1)
df['Group'] = split[0].astype(int)
df['Model'] = split[1]
df.set_index(['Target', 'Group', 'Model'], inplace=True)
df.sort_index(inplace=True)

Path('CASP11/QA_official').mkdir(exist_ok=True, parents=True)
df.to_pickle('CASP11/QA_official/table.pkl.xz')

print('Unique targets:', len(df.index.unique(level='Target')))
print('Unique groups:', len(df.index.unique(level='Group')))
df

# %% [markdown]
# ### CASP 12
# No .tar.gz file, only a list of .txt files

# %%
base_url = 'https://predictioncenter.org/download_area/CASP12/SUMMARY_TABLES/'
request = requests_get(base_url)
soup = bs4.BeautifulSoup(request.content)

dfs = {}
for a in soup.select('table tr a'):
    href = a.attrs['href']
    if regex.fullmatch(href):
        response = requests_get(base_url + href)
        df = pd.read_csv(io.BytesIO(response.content), sep='\s+')                
        dfs[href] = df
df = pd.concat(dfs.values(), axis=0)

# Split accession codes
split = df['Model'].str.split('TS', expand=True)
df['Target'] = split[0]
split = split[1].str.split('_', expand=True, n=1)
df['Group'] = split[0].astype(int)
df['Model'] = split[1]
df.set_index(['Target', 'Group', 'Model'], inplace=True)
df.sort_index(inplace=True)

Path('CASP12/QA_official').mkdir(exist_ok=True, parents=True)
df.to_pickle('CASP12/QA_official/table.pkl.xz')

print('Unique targets:', len(df.index.unique(level='Target')))
print('Unique groups:', len(df.index.unique(level='Group')))
df

# %% [markdown]
# ### CASP 13
# Single .tar.gz file with separate .txt files inside, 
# but also has some additional targets that are not in the .tar.gz file

# %%
url = 'https://predictioncenter.org/download_area/CASP13/results/tables/casp13.res_tables.T.tar.gz'
response = requests_get(url)
archive = tarfile.open(fileobj=io.BytesIO(response.content), mode='r')

dfs = {}
for member in archive.getmembers():
    if member.isfile() and regex.fullmatch(member.name):
        f = archive.extractfile(member)
        df = pd.read_csv(f, sep='\s+')                
        dfs[member.name] = df
df = pd.concat(dfs.values(), axis=0)

# Split accession codes
split = df['Model'].str.split('TS', expand=True)
df['Target'] = split[0]
split = split[1].str.split('_', expand=True, n=1)
df['Group'] = split[0].astype(int)
df['Model'] = split[1]
df.set_index(['Target', 'Group', 'Model'], inplace=True)
df.sort_index(inplace=True)

Path('CASP13/QA_official').mkdir(exist_ok=True, parents=True)
df.to_pickle('CASP13/QA_official/table.pkl.xz')

print('Unique targets:', len(df.index.unique(level='Target')))
print('Unique groups:', len(df.index.unique(level='Group')))
df

# %% [markdown]
# ### All CASPs together

# %%
df = pd.concat([
    pd.read_pickle(p)
    for p in Path().glob('CASP*/QA_official/table.pkl.xz')
], keys=[9,10,11,12,13], names=['Edition'])

print('Unique targets:', len(df.index.unique(level='Target')))
print('Unique groups:', len(df.index.unique(level='Group')))

with pd.option_context('display.max_rows', 15):
    cols_to_keep = ['GDT_TS', 'GDT_HA', 'LDDT', 'CAD_AA', 'TMscore']
    display(df[cols_to_keep])
    display(
        df.groupby('Edition')
            .apply(lambda df: df.index.get_level_values('Target').nunique())
            .to_frame('Targets')
    )
del df


# %% [markdown]
# ## Other group's QA predictions
#
# These are the QA predictions submitted by other participants to the QA track in CASP 
# ([file format reference](https://predictioncenter.org/casp13/index.cgi?page=format#QA)).
#
# Start with `PFRMAT QA`
#
# Use `MODEL 1` for predictions submitted in the first stage </br>
# (i.e., estimating quality of the selected server models released 5 days after the initial target release)
#
# Use `MODEL 2` for predictions submitted on the second, larger set of TS models </br>
# (i.e., estimating quality of models released 7 days after the initial target release).
#
# Timeline example.
# - May 1, 9am PDT - target T0644 is released for prediction in non-QA categories.
# - May 4, noon - the deadline for submitting tertiary structure predictions by servers.
# - May 6, noon - the first set of server TS predictions (up to 20 models selected primarily to test single-model methods) is sent to the registered QA servers and posted on the casp14 archive page (https://predictioncenter.org/download_area/CASP14/server_predictions/). QA predictions (marked as MODEL 1) for this subset are accepted for two days.
# - May 8, noon - deadline for "stage 1" QA predictions. The second set of server TS predictions (150 models selected to test both, single-model and clustering methods) is sent to the registered QA servers and posted on the casp14 archive page. QA predictions (marked as MODEL 2) for this second subset of models are accepted for two more days.
# - May 10, noon - deadline for "stage 2" QA predictions. All server TS predictions are posted on the casp14 archive page. No further QA predictions (from servers or manual groups) are accepted for this target.
#
# Data are inserted between `MODEL` and `END` records of the submission file. </br>
# You may submit your quality assessment prediction in one of the two different modes:
# - `QMODE 1` :   global model quality score (MQS - one number per model)
# - `QMODE 2` :   MQS and error estimates on per-residue basis.
#
# In both modes, the first column in each line contains model identifier (file name of the accepted 3D prediction). </br>
# The second column contains the accuracy score for a model as a whole (MQS). The accuracy score is a real number between 0.0 and 1.0 (1.0 being a perfect model). </br>
# If you don't provide error estimates on per residue basis, your data table will consist of these two columns only (Example A).
#
# If you do additionally provide residue error estimates (QMODE 2), 
# each consecutive column should contain error estimate in Angstroms for all the consecutive residues in the target 
# (i.e., column 3 corresponds to residue 1 in the target, column 4 - to residue 2 and so on). </br>
# This way data constitute a table (Number_of_models_for_the_target) BY (Number_of_residues_in_the_target + 1). </br>
# Do not skip columns if you are not predicting error estimates for some residues - instead put "X" in the corresponding column (Example B).</br>
# Please specify in the REMARKS what you consider to be an error estimate for a residue (CA location error, geometrical center error, etc.).
#
# Note 1. Please, be advised that a QA record line may be very long and that some editors/mailing programs may force line wrap potentially causing unexpected parsing errors. </br>
# To avoid this problem we recommend that you split long lines into shorter sublines (50-100 columns of data) by yourself. </br>
# Our parser will consider consecutive sublines (starting with the line containing evaluated model name and ending with the line containing the next model name or tag END) a part of the same logical line.
#
# Note 2. Please, be advised that model quality predictions in CASP are evaluated by comparing submitted estimates of 
# global reliability and per-residue accuracy of structural models with the values obtained from CASP model evaluation packages (LGA, LDDT, CAD-score and others). </br>
# Since the evaluation score that is used across the categories in CASP is GDT_TS, predictors should strive to predict this score in QMODE1 (QA1). </br>
# Predicted per-residue distances in QMODE2 should ideally reproduce those extracted from the LGA optimal model-target superpositions.
#
# Examples:
# - (A) Global Model Quality Score
#     ```
#     PFRMAT QA
#     TARGET T0999
#     AUTHOR 1234-5678-9000
#     METHOD Description of methods used
#     MODEL 1
#     QMODE 1
#     3D-JIGSAW_TS1 0.8 
#     FORTE1_AL1.pdb 0.7 
#     END
#     ```
# - (B) Residue-based Quality Assessment (fragment of the table). 
#   Note, that this case includes case (A) and there is no need to submit QMODE 1 predictions additionally to QMODE 2.
#     ```
#     PFRMAT QA
#     TARGET T0999
#     AUTHOR 1234-5678-9000
#     REMARK Error estimate is CA-CA distance in Angstroms
#     METHOD Description of methods used
#     MODEL 1
#     QMODE 2
#     3D-JIGSAW_TS1 0.8 10.0 6.5 5.0 2.0 1.0  
#     5.0 4.3 4.6
#     FORTE1_AL1.pdb 0.7 8.0 5.5 4.5 X X 
#     4.5 4.2 5.0 
#     END
#     ```

# %% [markdown]
# ### QA group names for CASP 11, 12, 13
#
# Each group participating in the QA track is assigned an id like `QA014`, 
# but we also need the name of the group, which can be easily reconnected to the QA method used by the group.

# %%
def extract_groups(soup):
    for tr in soup.select('tr'):
        if tr.attrs != {'class': [], 'onmouseover': 'row_over(this)', 'onmouseout': 'row_out(this)'}:
            continue
        tds = tr.select('td')
        if 'QA' not in tds[4].text:
            continue
        group_name = tds[0].text
        group_id = 'QA' + tds[1].text
        yield group_id, group_name

for casp_ed in [11,12,13]:
    response = requests_get(f'https://predictioncenter.org/casp{casp_ed}/docs.cgi?view=groupsbyname')
    soup = bs4.BeautifulSoup(response.content)
    
    df_groups = pd.DataFrame(
        list(extract_groups(soup)), 
        columns=['qa_group_id', 'qa_group_name']
    ).sort_values('qa_group_id').reset_index(drop=True)
    
    dest = Path(f'CASP{casp_ed}/QA_groups.csv')
    dest.parent.mkdir(exist_ok=True, parents=True)
    df_groups.to_csv(dest, header=True, index=False)

display(df_groups.style.set_caption('CASP 13').hide_index())

# %% [markdown]
# ### Download all QA predictions
#
# Each CASP has a slightly different file structure for QA predictions.

# %%
dest = 'CASP9/QA_predictions'
base_url = 'https://predictioncenter.org/download_area/CASP9/predictions/'
qa_urls = [
    'QA_T0515-T0539.tar.gz',
    'QA_T0540-T0569.tar.gz',
    'QA_T0570-T0599.tar.gz',
    'QA_T0600-T0629.tar.gz',
    'QA_T0630-T0643.tar.gz',
]

if not Path(dest).is_dir():
    Path(dest).mkdir(parents=True)
    for qa_url in qa_urls:
        # ! curl -s {base_url}{qa_url} | tar xz --directory {dest}
    for p in Path(dest).glob('T????QA???_?'):
        Path(dest).joinpath(p.name[:5]).mkdir(exist_ok=True)
        p.rename(Path(dest) / p.name[:5] / p.name)

# ! ls {dest} | wc -l
# ! du -sh {dest}

# %%
dest = 'CASP10/QA_predictions'
base_url = 'https://predictioncenter.org/download_area/CASP10/predictions/'
qa_urls = [
    'QA_T0644-T0669.tar.gz',
    'QA_T0670-T0699.tar.gz',
    'QA_T0700-T0729.tar.gz',
    'QA_T0730-T0758.tar.gz',
]

if not Path(dest).is_dir():
    Path(dest).mkdir(parents=True)
    for qa_url in qa_urls:
        # ! curl -s {base_url}{qa_url} | tar xz --strip 1 --directory {dest}

# ! ls {dest} | wc -l
# ! du -sh {dest}

# %%
dest = 'CASP11/QA_predictions'
base_url = 'https://predictioncenter.org/download_area/CASP11/predictions/'
qa_urls = [
    'QA_T0759-799.tar.gz',
    'QA_T0800-829.tar.gz',
    'QA_T0830-858.tar.gz',
]

if not Path(dest).is_dir():
    Path(dest).mkdir(parents=True)
    for qa_url in qa_urls:
        # ! curl -s {base_url}{qa_url} | tar xz --strip 1 --directory {dest}
        
# ! ls {dest} | wc -l
# ! du -sh {dest}

# %%
dest = 'CASP12/QA_predictions'
base_url = 'https://predictioncenter.org/download_area/CASP12/predictions/'
qa_urls = [
    'CASP12_QA_T08x.tgz',
    'CASP12_QA_T09x.tgz',
]

if not Path(dest).is_dir():
    Path(dest).mkdir(parents=True)
    for qa_url in qa_urls:
        # ! curl -s {base_url}{qa_url} | tar xz --directory {dest}
        
# ! ls {dest} | wc -l
# ! du -sh {dest}

# %%
# Map CASP 12 naming to decoy_id naming:
# T0949TS145_1 -> QUARK_TS1
# TODO

# %%
dest = 'CASP13/QA_predictions'
base_url = 'https://predictioncenter.org/download_area/CASP13/predictions/QA/'
qa_urls = [
    # Stage 1
    'QA1.all.tar.gz',
    # Stage 2
    'QA2.T095_.tar.gz',
    'QA2.T096_.tar.gz',
    'QA2.T097_.tar.gz',
    'QA2.T098_.tar.gz',
    'QA2.T099_.tar.gz',
    'QA2.T100_.tar.gz',
    'QA2.T101_.tar.gz',
    'QA2.T102_.tar.gz',
]

if not Path(dest).is_dir():
    Path(dest).mkdir(parents=True)
    for qa_url in qa_urls:
        # ! curl -s {base_url}{qa_url} | tar xz --directory {dest}

# ! ls {dest} | wc -l
# ! du -sh {dest}

# %%
# Map CASP 13 naming to decoy_id naming:
# T0949TS145_1 -> QUARK_TS1
mapping = {}

base_url = "https://www.predictioncenter.org/download_area/CASP13/predictions/TS_as_submitted/"
urls = [
   "T0949.TS_as_accepted.tar.gz",
   "T0950.TS_as_accepted.tar.gz",
   "T0951.TS_as_accepted.tar.gz",
   "T0953s1.TS_as_accepted.tar.gz",
   "T0953s2.TS_as_accepted.tar.gz",
   "T0954.TS_as_accepted.tar.gz",
   "T0955.TS_as_accepted.tar.gz",
   "T0956.TS_as_accepted.tar.gz",
   "T0957s1.TS_as_accepted.tar.gz",
   "T0957s2.TS_as_accepted.tar.gz",
   "T0958.TS_as_accepted.tar.gz",
   "T0959.TS_as_accepted.tar.gz",
   "T0960.TS_as_accepted.tar.gz",
   "T0961.TS_as_accepted.tar.gz",
   "T0962.TS_as_accepted.tar.gz",
   "T0963.TS_as_accepted.tar.gz",
   "T0964.TS_as_accepted.tar.gz",
   "T0965.TS_as_accepted.tar.gz",
   "T0966.TS_as_accepted.tar.gz",
   "T0967.TS_as_accepted.tar.gz",
   "T0968s1.TS_as_accepted.tar.gz",
   "T0968s2.TS_as_accepted.tar.gz",
   "T0969.TS_as_accepted.tar.gz",
   "T0970.TS_as_accepted.tar.gz",
   "T0971.TS_as_accepted.tar.gz",
   "T0972.TS_as_accepted.tar.gz",
   "T0973.TS_as_accepted.tar.gz",
   "T0974s1.TS_as_accepted.tar.gz",
   "T0974s2.TS_as_accepted.tar.gz",
   "T0975.TS_as_accepted.tar.gz",
   "T0976.TS_as_accepted.tar.gz",
   "T0977.TS_as_accepted.tar.gz",
   "T0978.TS_as_accepted.tar.gz",
   "T0979.TS_as_accepted.tar.gz",
   "T0980s1.TS_as_accepted.tar.gz",
   "T0980s2.TS_as_accepted.tar.gz",
   "T0981.TS_as_accepted.tar.gz",
   "T0982.TS_as_accepted.tar.gz",
   "T0983.TS_as_accepted.tar.gz",
   "T0984.TS_as_accepted.tar.gz",
   "T0985.TS_as_accepted.tar.gz",
   "T0986s1.TS_as_accepted.tar.gz",
   "T0986s2.TS_as_accepted.tar.gz",
   "T0987.TS_as_accepted.tar.gz",
   "T0988.TS_as_accepted.tar.gz",
   "T0989.TS_as_accepted.tar.gz",
   "T0990.TS_as_accepted.tar.gz",
   "T0991.TS_as_accepted.tar.gz",
   "T0992.TS_as_accepted.tar.gz",
   "T0993s1.TS_as_accepted.tar.gz",
   "T0993s2.TS_as_accepted.tar.gz",
   "T0994.TS_as_accepted.tar.gz",
   "T0995.TS_as_accepted.tar.gz",
   "T0996.TS_as_accepted.tar.gz",
   "T0997.TS_as_accepted.tar.gz",
   "T0998.TS_as_accepted.tar.gz",
   "T0999.TS_as_accepted.tar.gz",
   "T1000.TS_as_accepted.tar.gz",
   "T1001.TS_as_accepted.tar.gz",
   "T1002.TS_as_accepted.tar.gz",
   "T1003.TS_as_accepted.tar.gz",
   "T1004.TS_as_accepted.tar.gz",
   "T1005.TS_as_accepted.tar.gz",
   "T1006.TS_as_accepted.tar.gz",
   "T1007.TS_as_accepted.tar.gz",
   "T1008.TS_as_accepted.tar.gz",
   "T1009.TS_as_accepted.tar.gz",
   "T1010.TS_as_accepted.tar.gz",
   "T1011.TS_as_accepted.tar.gz",
   "T1012.TS_as_accepted.tar.gz",
   "T1013.TS_as_accepted.tar.gz",
   "T1014.TS_as_accepted.tar.gz",
   "T1015s1.TS_as_accepted.tar.gz",
   "T1015s2.TS_as_accepted.tar.gz",
   "T1016.TS_as_accepted.tar.gz",
   "T1017s1.TS_as_accepted.tar.gz",
   "T1017s2.TS_as_accepted.tar.gz",
   "T1018.TS_as_accepted.tar.gz",
   "T1019s1.TS_as_accepted.tar.gz",
   "T1019s2.TS_as_accepted.tar.gz",
   "T1020.TS_as_accepted.tar.gz",
   "T1021s1.TS_as_accepted.tar.gz",
   "T1021s2.TS_as_accepted.tar.gz",
   "T1021s3.TS_as_accepted.tar.gz",
   "T1022s1.TS_as_accepted.tar.gz",
   "T1022s2.TS_as_accepted.tar.gz",
   "T1023s1.TS_as_accepted.tar.gz",
   "T1023s2.TS_as_accepted.tar.gz",
   "T1023s3.TS_as_accepted.tar.gz",
]

for url in urls:
    target_dir = Path('/tmp/casp13').joinpath(url[:-7])
    if not target_dir.is_dir():
        target_dir.mkdir(parents=True)
        # ! curl -s {base_url}{url} | tar xz --strip 2 --directory {target_dir.as_posix()}
    
    for p in target_dir.iterdir():
        try:
            with p.open() as f:
                for l in f:
                    if l.startswith('TARGET'):
                        target_id = l.split()[1]
                    if l.startswith('AUTHOR'):
                        decoy_id = l.split()[1]
                    if l.startswith('MODEL'):
                        model = l.split()[1]
                        break
                decoy_id = f'{decoy_id}_TS{model}'
                mapping[p.name] = (target_id, decoy_id)
                del target_id, decoy_id, model
        except UnicodeDecodeError as e:
            print(p, e)
        except IndexError as e:
            print(p, repr(l), e)
        except ValueError as e:
            print(p, repr(l), e)
            
# ! rm -r '/tmp/casp13'
with open('CASP13/decoy_name_mapping.pkl', 'wb') as f:
    pickle.dump(mapping, f)


# %% [markdown]
# ### Parsing QA submissions
#
# These are some QA submissions for target `T0759`:

# %%
# ! ls CASP11/QA_predictions/T0759 | head -n3

# %% [markdown]
# These are the predictions made by QA group `QA008` for all decoys of target `T0759` submitted in stage `1`:

# %%
# ! head -n 10 CASP11/QA_predictions/T0759/T0759QA008_1

# %% [markdown]
# These are the predictions made by QA group `QA008` for all decoys of target `T0759` submitted in stage `2`:

# %%
# ! head -n 10 CASP11/QA_predictions/T0759/T0759QA008_2

# %%
def parse_float_score(score):
    if score == 'X':
        return float('NaN')
    return float(score)

def parse_filename(path):
    target, rest = path.name.split('QA')
    qa_group, stage = rest.split('_')
    return target, f'QA{qa_group}', int(stage)

@logger.catch(reraise=True)
def parse_qa_submission(path):   
    with open(path) as f:
        pformat = f.readline().split()
        if pformat != ['PFRMAT', 'QA']:
            raise ValueError(pformat)
        target = f.readline().split()[1]
        stage = int(f.readline().split()[1])
        qmode = int(f.readline().split()[1])

        if qmode == 1:
            split = f.readline().split()
            while split[0] != 'END':
                decoy, global_score = split
                yield decoy, parse_float_score(global_score), None
                split = f.readline().split()
        elif qmode == 2:
            decoy = None
            global_score = None
            local_scores = None

            line = f.readline()
            while line != '' and line.strip() != 'END':
                split = line.split()
                decoy = split[0]
                global_score = parse_float_score(split[1])
                local_scores = [parse_float_score(s) for s in split[2:]]

                try:
                    while line != '':
                        line = f.readline()
                        split = line.split()
                        local_scores.extend(parse_float_score(s) for s in split)
                except ValueError:
                    pass
                yield decoy, global_score, local_scores
        else:
            raise ValueError(qmode)


# %%
for casp_ed in [11,12,13]: # [9,10,11,12,13]
    df_global = []
    df_local = {}
    qa_group_names = pd.read_csv(f'CASP{casp_ed}/QA_groups.csv').set_index('qa_group_id')['qa_group_name'].to_dict()
    
    if casp_ed == 13:
        with open('CASP13/decoy_name_mapping.pkl', 'rb') as f:
            casp_13_decoy_mapping = pickle.load(f)
    
    parsed_decoys = 0
    targets = list(Path(f'CASP{casp_ed}/QA_predictions').glob('T*'))
    casp_bar = tqdm.tqdm(targets, desc=f'CASP{casp_ed}', unit='targets')
    for target_path in casp_bar:
        for path in target_path.glob('T*QA*'):
            target_id, qa_group_id, stage = parse_filename(path)
            qa_group_id = qa_group_names[qa_group_id]
            
            for decoy_id, global_score, local_scores in parse_qa_submission(path):
                if casp_ed == 13:
                    _, decoy_id = casp_13_decoy_mapping[decoy_id]
                parsed_decoys += 1
                df_global.append((qa_group_id, target_id, decoy_id, stage, global_score))
                if local_scores is not None:
                    local_scores = pd.Series(local_scores, name='pred').rename_axis('residue_idx')
                    df_local[(qa_group_id, target_id, decoy_id, stage)] = local_scores
        
        casp_bar.set_postfix({'parsed decoys': parsed_decoys})
    casp_bar.close()
    
    # Global scores
    df_global = pd.DataFrame(df_global, columns=['qa_group_id', 'target_id', 'decoy_id', 'stage', 'pred'])
    df_global.sort_values(['qa_group_id', 'target_id', 'decoy_id', 'stage', 'pred'], inplace=True)
    (
        df_global
        .set_index(['qa_group_id', 'target_id', 'decoy_id', 'stage'])
        .to_pickle(f'CASP{casp_ed}/QA_predictions/global.pkl')
    )
    
    print('Raw dataframe')
    display(df_global.set_index(['qa_group_id', 'target_id', 'decoy_id', 'stage']))

    print('Number of decoys of each target scored by each group in each stage')
    display(
        df_global.groupby(['qa_group_id', 'target_id', 'stage'])
        .size()
        .unstack('stage', fill_value=0)
    )

    print('Number of targets considered by each group in each CASP')
    display(
        df_global
        .groupby(['qa_group_id', 'stage'])
        .agg({'target_id': 'nunique'})
        .unstack('stage', fill_value=0)
    )
    del df_global
    
    # Local scores
    df_local = {k: df_local[k] for k in sorted(df_local.keys())}
    df_local = pd.concat(
        df_local.values(), 
        keys=df_local.keys(), 
        names=['qa_group_id', 'target_id', 'decoy_id', 'stage']
    ).to_frame('pred')
    df_local.to_pickle(f'CASP{casp_ed}/QA_predictions/local.pkl')
    
    print('Raw dataframe')
    display(df_local)
    
    print('By stage')
    display(
        df_local
        .groupby(['qa_group_id', 'target_id', 'decoy_id', 'stage'])
        .first()
        .groupby(['qa_group_id', 'target_id', 'stage'])
        .size()
        .unstack('stage', fill_value=0)
    )
    del df_local

# %% language="bash"
# # Global QA predictions are small enough
# du -hsc CASP*/QA_predictions/global.pkl
# echo
#
# # Local QA predictions should be compressed
# du -hsc CASP*/QA_predictions/local.pkl
# for f in CASP*/QA_predictions/local.pkl; do
#   xz -1 --compress "$f"
# done
# echo
#
# du -hsc CASP*/QA_predictions/local.pkl.xz

# %% [markdown]
# ## Official QA local scores
#
# Offical local scores (per-residue) are only avaliable for CASP 13.

# %% [markdown]
# ### CASP 13

# %%
base_url = 'https://predictioncenter.org/download_area/CASP13/results/sda/'
response = requests_get(base_url)
soup = bs4.BeautifulSoup(response.content)

links = [
    a.attrs['href']
    for a in soup.select('tr td:nth-child(2) a')
    if re.search(r'T\d{4}(?:s\d)?.*\.tgz', a.attrs['href'])
]
with tempfile.NamedTemporaryFile(mode='w') as f:
    f.write('\n'.join([base_url + l for l in links]))
    # ! wget --quiet --input-file {f.name} --directory-prefix='CASP13/QA_official'

for l in links:
    # ! tar xf "CASP13/QA_official/{l}" --directory 'CASP13/QA_official/' && rm "CASP13/QA_official/{l}"

# ! du -sh CASP13/QA_official/

# %%
with open('CASP13/decoy_name_mapping.pkl', 'rb') as f:
    decoy_name_mapping = pickle.load(f)

def parse_lga(lga_path):
    residue_dist_mapping = {}
    with open(lga_path) as f:
        for l in filter(lambda l: l.startswith('LGA '), f):
            residue_idx = int(l.split()[2]) - 1
            distance = float(l.split()[5])
            residue_dist_mapping[residue_idx] = distance
    return residue_dist_mapping


# %%
target_links_mapping = {}
for l in links:
    target_id = l.split('.')[0].split('-')[0]
    target_links_mapping.setdefault(target_id, []).append(l)

for target_id in target_links_mapping:
    if f'{target_id}.tgz' in target_links_mapping[target_id]:
        # T0950.tgz is avalable, ignore T0950-D?.tgz files
        target_links_mapping[target_id] = f'{target_id}.tgz'
        
distances_true = {
    # target_id -> {
    #     decoy_id -> {
    #        residue_idx -> distance
    #     }
    # }
}
        
for target_id in target_links_mapping:
    target_series = {}
    if isinstance(target_links_mapping[target_id], str):
        # There is a single T0950.tgz file, load all ground-truth distances from it
        for lga_file in Path(f'CASP13/QA_official/{target_links_mapping[target_id]}').with_suffix('').glob('*.lga'):
            try:
                _, decoy_id = decoy_name_mapping[lga_file.with_suffix('').name]
            except KeyError:
                logger.warning(f'{lga_file.with_suffix("").name} not found in decoy_name_mapping')
                continue
            lga_dict = parse_lga(lga_file)
            if len(lga_dict) == 0:
                logger.warning(f'Check {lga_file} ({target_id} {decoy_id})')
                continue
            decoy_series = pd.Series(lga_dict).rename_axis('residue_idx').rename('true')
            target_series[decoy_id] = decoy_series
    else:
        # The target has been split into domains, e.g. T0984-D1.tgz T0984-D2.tgz, 
        # must merge individual files
        for domain_folder in target_links_mapping[target_id]:
            for lga_file in Path(f'CASP13/QA_official/{domain_folder}').with_suffix('').glob('*.lga'):
                try:
                    _, decoy_id = decoy_name_mapping[lga_file.with_suffix('').name.split('-')[0]]
                except KeyError:
                    logger.warning(f'{lga_file.with_suffix("").name} not found in decoy_name_mapping')
                    continue
                lga_dict = parse_lga(lga_file)
                if len(lga_dict) == 0:
                    logger.warning(f'Check {lga_file} ({target_id} {decoy_id})')
                    continue
                target_series.setdefault(decoy_id, {}).update(lga_dict)
        for decoy_id in target_series:
            target_series[decoy_id] = pd.Series(target_series[decoy_id]).rename_axis('residue_idx').rename('true')
    
    # Concat all decoys of a target into a single series
    distances_true[target_id] = pd.concat(
        [v for v in target_series.values() if isinstance(v, pd.Series)], 
        keys=[k for k in target_series if isinstance(target_series[k], pd.Series)], 
        names=['decoy_id']
    )

# Concat all targets of CASP13 into a single series    
distances_true = pd.concat(distances_true.values(), keys=distances_true.keys(), names=['target_id'])
                
distances_true.to_pickle('CASP13/QA_official/distances_true.pkl')
# ! du -h 'CASP13/QA_official/distances_true.pkl'

# %%
# Example dataframe
distances_true.to_frame()
