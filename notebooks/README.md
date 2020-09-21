# CASP Data Download and Preprocessing

## Overview of notebooks

Main notebooks:
1. [DownloadCaspData](./01-DownloadCaspData.ipynb): 
   download raw data from the CASP website
2. [DsspSecondaryStructure](./02-DsspSecondaryStructure.ipynb): 
   compute DSSP features
3. [MultipleSequenceAlignment](./03-MultipleSequenceAlignment.ipynb): 
   compute MSA features
4. [CadScore](./04-CadScore.ipynb): 
   compute ground-truth CAD scores
5. [LddtScore](./05-LddtScore.ipynb): 
   compute ground-truth LDDT scores
6. [TmScore](./06-TmScore.ipynb): 
   compute ground-truth GDT-TS and TM scores
7. [Preprocessing](./07-Preprocessing.ipynb):
   compute graph features and repack all input features and ground-truth scores for training
8. [Training](./08-Training.ipynb):
   an example training script (actual one is in [`src/graphqa/train.py`](../src/graphqa/train.py))
9. [QaMetrics](./09-QaMetrics.ipynb):
   compute QA metrics by comparing predicted and ground-truth scores

Other notebooks and files:
- [GraphConnectivityAndSeparationEncoding](./GraphConnectivityAndSeparationEncoding.ipynb):
  examples of graph connectivity by distance and separation
- [PositionalEncoding](./PositionalEncoding.ipynb):
  simple implementation of positional encoding
- [ProteinMetrics](./ProteinMetrics.ipynb):
  difference between all-models and per-model Pearson correlation of local scores
- [RankingMetrics](./RankingMetrics.ipynb):
  recall@x and normalized cumulative discount
- [Zscore](./Zscore.ipynb):
  example z-score calculation

[DownloadCaspData](./01-DownloadCaspData.ipynb) is the main notebook for downloading the raw protein data from the CASP website.
This notebook should be run first if one wishes to reproduce the experiments in the paper.
 
All other notebooks represent an easy-to-follow overview of the entire pre- and post-processing pipeline, 
but might not reflect exactly the steps used in the experiments. 
The actual pre- and post-processing code, rewritten for efficiency and ease-of-use, can be found in 
[`src/graphqa/data`](../src/graphqa/data).

## External software

We use the following external tools for pre-processing:
- [Jackhmmer (v3.3)](http://hmmer.org/documentation.html) for multiple-sequence alignments
- [Dockerized DSSP (697deab)](https://github.com/cmbi/dssp) for computing DSSP
- [Voronota (v1.21.2744)](https://kliment-olechnovic.github.io/voronota/) for computing CAD scores
- [Dockerized OpenStructure (v2.1.0)](https://git.scicore.unibas.ch/schwede/openstructure/-/tree/master/docker) for computing LDDT scores
- [TMscore (v2019/11/25)](https://zhanglab.ccmb.med.umich.edu/TM-score/) for computing TM scores

## Folder structure

For each protein dataset, the following folder structure is used:
```
CASP13
├── sequences.fasta                    Primary sequences
├── alignments.pkl                     Multiple-sequence alignment features
├── QA_groups.csv                      QA group names and ids
├── decoy_name_mapping.pkl             Mapping from decoy filenames to (target, decoy) pairs, 
│                                      e.g. 'T0953s1TS368_3' -> ('T0953s1', 'BAKER-ROSETTASERVER_TS3')
├── decoys                             Decoy files (raw structures, dssp outputs, ground-truth scores)
│   ├── T0949.cad.npz
│   ├── T0949.lddt.npz
│   ├── T0949.tmscore.npz
│   ├── T0949
│   |   ├── 3D-JIGSAW_SL1_TS1.dssp
│   |   ├── 3D-JIGSAW_SL1_TS1.pdb
│   |   ├── ...
|   |   ├── Zhou-SPOT-3D_TS5.dssp
|   |   └── Zhou-SPOT-3D_TS5.pdb
│   ├── ...
│   ├── T1022s2.cad.npz
│   ├── T1022s2.lddt.npz
│   ├── T1022s2.tmscore.npz
│   └── T1022s2
│       ├── 3D-JIGSAW_SL1_TS1.dssp
│       ├── 3D-JIGSAW_SL1_TS1.pdb
│       ├── ...
|       ├── Zhou-SPOT-3D_TS5.dssp
|       └── Zhou-SPOT-3D_TS5.pdb
├── native                             Native structures
│   ├── T0949.pdb
│   ├── ...
│   └── T1022s2.pdb
├── processed                          Decoys processed for training (precomputed graphs)
│   ├── T0949.pth
│   ├── ...
│   └── T1022s2.pth
├── QA_official                        Official local QA scores from CASP
│   ├── T0949
│   │   ├── T0949QA014_1.lga
│   │   ├── T0949QA014_2.lga
│   │   ├── ...
│   |   ├── T1022s2QA471_1.lga
│   |   └── T1022s2QA471_2.lga
│   ├── ...
│   └── T1022s2
│       ├── T1022s2QA014_1.lga
│       ├── T1022s2QA014_2.lga
│       ├── ...
│       ├── T1022s2QA471_1.lga
│       └── T1022s2QA471_2.lga
└── QA_predictions                     QA predictions made by other groups (CASP QA format)
    ├── T0949
    │   ├── T0949QA014_1
    │   ├── T0949QA014_2
    │   ├── ...
    |   ├── T1022s2QA471_1
    |   └── T1022s2QA471_2
    ├── ...
    └── T1022s2
        ├── T1022s2QA014_1
        ├── T1022s2QA014_2
        ├── ...
        ├── T1022s2QA471_1
        └── T1022s2QA471_2          
```

## Notes

All notebooks are saved as `.ipynb` and `.py` (percent script) and kept in sync through [Jupytext](https://github.com/mwouts/jupytext).
