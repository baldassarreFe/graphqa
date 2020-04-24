# Protein quality

## Initial setup
Clone repository, install package and prepare directory structure:
```bash
git clone https://github.com/baldassarreFe/graphqa
cd graphqa

export PATH="/usr/local/cuda/bin:${PATH}"
export CPATH="/usr/local/cuda/include:${CPATH}"
conda env create -n graphqa -f conda.yaml
conda activate graphqa
pip install .
```

Download and preprocess the data:
```bash
data/download-datasets.sh
for f in data/*.h5; do
    python -m proteins.dataset preprocess --filepath "${f}" --destpath "${f%.h5}" [--compress]
done
```

Tensorboard plugins: layout of custom scalars and hyper parameters
```bash
python -m proteins.layout runs/
python -m proteins.hparams runs/
```

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
