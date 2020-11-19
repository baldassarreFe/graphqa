# Prediction

Requirements:
- DSSP docker image:
  ```bash
  git clone https://github.com/cmbi/dssp ~/dssp
  pushd ~/dssp
  git checkout 697deab74011bfbd55891e9b8d5d47b8e4ef0e38
  docker build -t dssp .
  popd
  ```
- Uniref50 database:
  ```bash
  wget -q 'ftp://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz' 
  gzip --decompress 'uniref50.fasta.gz'
  rm 'uniref50.fasta.gz'
  ```
- A GraphQA checkpoint, for example [this checkpoint](./runs/wandb/quirky_stallman_1133/checkpoints/graphqa_epoch=599.ckpt)
  
## Prediction 

Input files:
- Sequences in Fasta format in file called `sequences.fasta`:
  ```fasta
  >T1234
  ACKIENI.......RFDLLTR
  
  >T5678
  NDVKKQQ.......GAPVPKQ
  ```
- Decoys in PDB format in a `decoys` folder with this structure (note the `.pdb` extension):
  ```
  decoys
  ├── T1234
  │   ├── first_decoy.pdb
  │   ├── another_decoy.pdb
  │   └── third_decoy.pdb
  └── T5678
      ├── first_decoy.pdb
      ├── another_decoy.pdb
      └── third_decoy.pdb
  ```
  
Preprocess the decoys:
```
UNIREF_50='path/to/uniref50.fasta'
python -m graphqa.data.preprocess . "${UNIREF_50}"

2020-11-19 11:05:30 | INFO | graphqa.data.dssp:run_dssp:239 - Running on 6 pdb files
2020-11-19 11:05:32 | INFO | graphqa.data.msa:run_msa:146 - Running on 2 sequence(s)
2020-11-19 11:18:12 | INFO | __main__:main:121 - Processing T1234
2020-11-19 11:18:12 | INFO | __main__:main:121 - Processing T5678
```

This will create the following files:
```
decoys
├── T1234
│   ├── first_decoy.pdb
│   ├── first_decoy.dssp
│   ├── another_decoy.pdb
│   ├── another_decoy.dssp
│   └── third_decoy.pdb
│   └── third_decoy.dssp
└── T5678
  ├── first_decoy.pdb
  ├── first_decoy.dssp
  ├── another_decoy.pdb
  ├── another_decoy.dssp
  └── third_decoy.pdb
  └── third_decoy.dssp

processed
├── T1234.pth
└── T5678.pth
```

Run GraphQA:
```bash
STAGE=1
CHECKPOINT='path/to/graphqa.ckpt'
python -m graphqa.eval "${CHECKPOINT}" . --stage "${STAGE}"

2020-11-19 11:21:59 | DEBUG | graphqa.dataset:__init__:25 - Starting to load graphs from 2 pth files
2020-11-19 11:21:59 | DEBUG | graphqa.dataset:__init__:69 - Done loading 6 graphs, skipped 0
```

This will create the following files:
```
predictions
├── global.csv
├── local.csv
├── T1234.stage1.qa
└── T5678.stage1.qa
```

The file `global.csv` contains all decoy-level predictions in this format:

| target_id | decoy_id      | tm   | gdtts | gdtha | lddt | cad  |
|-----------|---------------|------|-------|-------|------|------|
| T1234     | first_decoy   | 0.53 | 0.45  | 0.29  | 0.46 | 0.31 |
| T1234     | another_decoy | 0.29 | 0.46  | 0.31  | 0.53 | 0.45 |
| ...       | ...           | ...  | ...   | ...   | ...  | ...  |

The file `local.csv` contains all residue-level predictions in this format:

| target_id | decoy_id    | residue_idx | lddt | cad  |
|-----------|-------------|-------------|------|------|
| T1234     | first_decoy | 0           | 0.47 | 0.26 |
| T1234     | first_decoy | 1           | 0.54 | 0.39 |
| T1234     | first_decoy | 2           | 0.50 | 0.29 |
| ...       | ...         | ...         | ...  | ...  |
| T5678     | third_decoy | 268         | 0.47 | 0.26 |
| T5678     | third_decoy | 269         | 0.54 | 0.39 |
| T5678     | third_decoy | 270         | 0.50 | 0.29 |

Note that `residue_idx` is 0-indexed.

## CASP emails

CASP emails come in a standard format that can be easily processed. 

We assume the file `emails.txt` contains some emails in this format:
```
TARGET=T1031
SEQUENCE=ACKIE...LLTR
REPLY_EMAIL=models@predictioncenter.org
STOICHIOMETRY=A1
TARBALL_LOCATION=http://predictioncenter.org/.../T1031.stage1.3D.srv.tar.gz

TARGET=T1032
SEQUENCE=NDVKKQ...LRGMVFGAPVPKQ
REPLY_EMAIL=models@predictioncenter.org
STOICHIOMETRY=A2
TARBALL_LOCATION=http://predictioncenter.org/.../T1032.stage1.3D.srv.tar.gz
```

Download all decoys and extract the tar files:
```bash
grep 'TARBALL_LOCATION=' 'emails.txt' | while read -r line ; do
  curl -L "${line/TARBALL_LOCATION=/}" | tar xz --directory 'decoys'
done
```

Add `.pdb` to each decoy file:
```bash
for f in decoys/*/*; do
  mv "$f" "${f%.pdb}.pdb"
done
```

Extract sequences in fasta format:
```bash
cat 'emails.txt' |
  sed '/REPLY_EMAIL=/d' | 
  sed '/STOICHIOMETRY=/d' | 
  sed '/TARBALL_LOCATION=/d' | 
  sed 's/TARGET=/>/' | 
  sed 's/SEQUENCE=//' > sequences.fasta
```

Run GraphQA:
```bash 
conda activate proteins

STAGE=1
CHECKPOINT='path/to/graphqa.ckpt'
UNIREF_50='path/to/uniref50.fasta'

python -m graphqa.data.preprocess . "${UNIREF_50}"
python -m graphqa.eval "${CHECKPOINT}" . --stage "${STAGE}"
```
