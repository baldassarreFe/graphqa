# GraphQA: Protein Model Quality Assessment using Graph Convolutional Networks

## Evaluation server
Try it yourself!
A simple implementation of an evaluation server is available at this [link](http://isengard.csc.kth.se:8585/).

## Initial setup
Clone repository, install dependencies in a conda environment, install GraphQA:
```bash
git clone https://github.com/baldassarreFe/graphqa
cd graphqa

export PATH="/usr/local/cuda/bin:${PATH}"
export CPATH="/usr/local/cuda/include:${CPATH}"
conda env create -n graphqa -f conda.yaml
conda activate graphqa
pip install .
```

## Prediction

To make predictions using GraphQA, follow the instructions in [`predictions.md`](./prediction.md).

## Datasets

### Manual download and preprocessing
The file [`notebooks/README.md`](./notebooks/README.md) contains all information 
to download and preprocess CASP data for training GraphQA. At a high level, the necessary steps are:
1. Download protein sequences, official native structures, submitted decoy structures, 
   submitted QA predictions, and official QA scores from the CASP website
2. Run DSSP on all submitted tertiary structures to extract secondary structure features
3. Run JackHMMER on all protein sequences to compute multiple-sequence alignment features against UniRef50
4. Score all decoys with respect to the respective native structures, specifically computing: 
   - per-residue: CAD and LDDT scores
   - per-decoy: GDT_TS, GDT_TS, TM, CAD, LDDT scores
5. Transform each decoy into a graph data structure suitable for training with PyTorch, including 
   all input and output features computed in the steps above. At this stage, geometric and sequential 
   features are also added to the graph (edges, distances and angles) to avoid computing them during training.  

First, run the [DownloadCaspData notebook](./notebooks/01-DownloadCaspData.ipynb) to download 
raw protein data from the CASP website.

Then, prepare all preprocessing tools (some of them require a compilation step, others run in Docker): 
```bash
# Docker image for DSSP
docker build -t dssp 'https://github.com/cmbi/dssp.git#697deab74011bfbd55891e9b8d5d47b8e4ef0e38'

# Sequence database for JackHMMER
wget 'ftp://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz'
tar xzf 'uniref50.fasta.gz'

# Docker image for LDDT score
docker pull 'registry.scicore.unibas.ch/schwede/openstructure:2.1.0'

# Voronota binaries for CAD score
wget 'https://github.com/kliment-olechnovic/voronota/releases/download/v1.21.2744/voronota_1.21.2744.tar.gz'
tar xzf 'voronota_1.21.2744.tar.gz'

# TMscore source for GDT_TS, GDT_HA, TM scores
wget 'https://zhanglab.ccmb.med.umich.edu/TM-score/TMscore.cpp'
g++ -static -O3 -ffast-math -lm -o TMscore TMscore.cpp 
```

Run preprocessing for training:
```bash
for CASP in data/CASP{9..13}; do
  python -m graphqa.data.preprocess "$CASP" "uniref50.fasta" \
    --train \
    --tmscore "./TMscore" \
    --voronota "./voronota_1.21.2744/voronota-cadscore"
done
```

### Download preprocessed data
Downloading the data and running the preprocessing steps described above can take a long time.
To skip these steps and directly download the dataset used for training:
```bash
BASE_URL='https://kth.box.com/shared/static/'
wget -O GraphQA-CASP9.tar.gz  "${BASE_URL}fm2weje86d7nvulbconzf3pzmmhl2tmm.gz"
wget -O GraphQA-CASP10.tar.gz "${BASE_URL}jdgns10ehenjur1y5dw2lj275aggeu33.gz"
wget -O GraphQA-CASP11.tar.gz "${BASE_URL}tls5yxhsycqpid8pp6i3jv7ew7h0xz6l.gz"
wget -O GraphQA-CASP12.tar.gz "${BASE_URL}cbm3k5ladnq5i42q5fdcbztxwaukde9x.gz"
wget -O GraphQA-CASP13.tar.gz "${BASE_URL}f66fjw67urwxcovfrpar5jd4diyayshl.gz"
```

Extract the contents of the tar archives in the corresponding folders under `/data`.

## Training
Either train with a predefined configuration
```bash
python -m proteins.train config/train.yaml --model config/model.yaml --session config/session.yaml [in_memory=yes]
```

Or define all parameters manually
```bash
# Data
cutoff=10
partial_entropy=no
self_information=no
dssp=no

# Model
model_fn=proteins.networks.ProteinGN
layers=6
min_dist=0
max_dist=20
rbf_size=16
residue_emb_size=64
separation_enc=categorical
distance_enc=rbf
mp_in_edges=128
mp_in_nodes=512
mp_in_globals=512
mp_out_edges=16
mp_out_nodes=64
mp_out_globals=32
dropout=.2
batch_norm=no

# Losses
loss_local_lddt=5
loss_global_gdtts=5

# Optimizer
opt_fn=torch.optim.Adam
learning_rate=.001
weight_decay=.00001

# Session
max_epochs=10
batch_size=1000
datasets='[data/CASP7,data/CASP8,data/CASP9,data/CASP10]'
logs='~/proteins/runs'

tags=()
tags+=("residueonly")
tags+=("l${layers}")
tags+=("${mp_in_edges}-${mp_in_nodes}-${mp_in_globals}")
tags+=("${mp_out_edges}-${mp_out_nodes}-${mp_out_globals}")
tags+=("dr${dropout}")
tags+=("bn${batch_norm}")
tags+=("lr${learning_rate}")
tags+=("wd${weight_decay}")
tags+=("ll${loss_local_lddt}")
tags+=("lg${loss_global_gdtts}")
tags+=("co${cutoff}")
tags+=("res${residue_emb_size}")
tags+=("rbf${rbf_size}")
tags+=("sep${separation_enc}")
tags+=("dist${distance_enc}")
tags="[$(IFS=, ; echo "${tags[*]}")]"

python -m proteins.train \
    tags="${tags}" \
    --data \
        cutoff="${cutoff}" \
        partial_entropy="${partial_entropy}" \
        self_information="${self_information}" \
        dssp="${dssp}" \
    --model \
        fn="${model_fn}" \
        layers="${layers}" \
        dropout="${dropout}" \
        batch_norm="${batch_norm}" \
        min_dist="${min_dist}" \
        max_dist="${max_dist}" \
        rbf_size="${rbf_size}" \
        residue_emb_size="${residue_emb_size}" \
        separation_enc="${separation_enc}" \
        distance_enc="${distance_enc}" \
        mp_in_edges="${mp_in_edges}" \
        mp_in_nodes="${mp_in_nodes}" \
        mp_in_globals="${mp_in_globals}" \
        mp_out_edges="${mp_out_edges}" \
        mp_out_nodes="${mp_out_nodes}" \
        mp_out_globals="${mp_out_globals}" \
    --loss.local_lddt \
        name=mse \
        weight="${loss_local_lddt}" \
    --loss.global_gdtts \
        name=mse \
        weight="${loss_global_gdtts}" \
    --optimizer \
        fn="${opt_fn}" \
        lr="${learning_rate}" \
        weight_decay="${weight_decay}" \
    --session.data \
        trainval="${datasets}" \
        split=35 \
        in_memory=yes \
    --session.logs \
        folder="${logs}" \
    --session \
        cpus=1 \
        checkpoint=2 \
        max_epochs="${max_epochs}" \
        batch_size="${batch_size}"
```

Logs and checkpoints can be found in `runs`:
```bash
tensorboard --logdir runs
```

## Ablation studies
Config files for ablation studies are self-contained and can just be run as:
```bash
NUM_RUNS_PER_STUDY=5
for f in config/ablations/{nodes,edges,layersvscutoff,architecture,localglobalscore,separation_encoding}/*.yaml; do
    for i in $(seq ${NUM_RUNS_PER_STUDY}); do
        python -m proteins.train "${f}"
    done
done
```

## Testing
Test GraphQA with all features (residues, multiple-sequence alignment, DSSP):
```bash
RUN_PATH='runs/l6_128-512-512_16-64-32_res64_rbf32_sepcategorical_dr.2_bnno_lr.001_wd.00001_ll1_lg1_lr0_co8_allfeats_wonderful_mclean'
for data in $(find 'data/' -maxdepth 1 -mindepth 1 -type d); do
    python -m proteins.test \
      "${RUN_PATH}/experiment.latest.yaml" \
      --model state_dict="${RUN_PATH}/model.latest.pt" \
      --test \
        data.input="${data}" \
        data.output="results/allfeatures/$(basename "${data}")" \
        data.in_memory=yes \
        cpus=1 \
        batch_size=200 
done
```

Test GraphQA with residue identity features only:
```bash
RUN_PATH='runs/residueonly_l8_128-512-512_16-64-64_dr.1_bnno_lr.001_wd.00001_ll1_ll5_co8_priceless_hawking'
for data in $(find 'data/' -maxdepth 1 -mindepth 1 -type d); do
    python -m proteins.test \
      "${RUN_PATH}/experiment.latest.yaml" \
      --model state_dict="${RUN_PATH}/model.latest.pt" \
      --test \
        data.input="${data}" \
        data.output="results/residueonly/$(basename "${data}")" \
        data.in_memory=yes \
        cpus=1 \
        batch_size=200 
done
```
