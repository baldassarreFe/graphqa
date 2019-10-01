# Protein quality

## Initial setup
Clone repository, install package and prepare directory structure:
```bash
git clone https://github.com/baldassarreFe/protein-quality-gn
cd protein-quality-gn

export PATH="/usr/local/cuda/bin:${PATH}"
export CPATH="/usr/local/cuda/include:${CPATH}"
conda env create -n proteins -f conda.yaml
conda activate proteins
pip install .
```

Download and preprocess the data:
```bash
data/download-datasets.sh
for f in data/*.h5; do
    python -m proteins.dataset preprocess --filepath "${f}" --destpath "${f%.h5}" 
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
python -m proteins.train config/train.yaml --model config/model.yaml --session config/session.yaml
```

Or define all parameters manually
```bash
# Data
cutoff=8
sigma=15
separation=yes

# Model
layers=6
dropout=.2
batch_norm=no
mp_in_edges=128
mp_in_nodes=512
mp_in_globals=512
mp_out_edges=16
mp_out_nodes=64
mp_out_globals=32

# Losses
loss_local_lddt=5
loss_global_gdtts=5

# Optimizer
learning_rate=.001
weight_decay=.00001

# Session
max_epochs=10
batch_size=200
datasets='[data/CASP7,data/CASP8,data/CASP9,data/CASP10]'

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
tags+=("si${sigma}")
tags+=("se${separation}")
tags="[$(IFS=, ; echo "${tags[*]}")]"

python -m proteins.train \
    tags="${tags}" \
    --data \
        cutoff="${cutoff}" \
        sigma="${sigma}" \
        separation="${separation}" \
        encoding_size=0 \
        encoding_base=0 \
        residues=yes \
        partial_entropy=no \
        self_info=no \
        dssp_features=no \
    --model \
        fn=proteins.networks.ProteinGN \
        layers="${layers}" \
        dropout="${dropout}" \
        batch_norm="${batch_norm}" \
        mp_in_edges="${mp_in_edges}" \
        mp_in_nodes="${mp_in_nodes}" \
        mp_in_globals="${mp_in_globals}" \
        mp_out_edges="${mp_out_edges}" \
        mp_out_nodes="${mp_out_nodes}" \
        mp_out_globals="${mp_out_globals}" \
    --loss.local_lddt \
        name=mse \
        weight="${loss_local_lddt}" \
        balanced=no \
    --loss.global_lddt \
        name=mse \
        weight=0 \
        balanced=no \
    --loss.global_gdtts \
        name=mse \
        weight="${loss_global_gdtts}" \
        balanced=no \
    --optimizer \
        fn=torch.optim.Adam \
        lr="${learning_rate}" \
        weight_decay="${weight_decay}" \
    --session \
        data.trainval="${datasets}" \
        data.split=35 \
        cpus=8 \
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
for f in config/ablations/{nodes,edges,layersvscutoff,architecture,localglobalscore}/*.yaml; do
    for i in $(seq ${NUM_RUNS_PER_STUDY}); do
        python -m proteins.train "${f}"
    done
done
```

## Testing
Test GraphQA with all features (residues, multiple-sequence alignment, DSSP):
```bash
RUN_PATH='runs/l6_128-512-512_16-64-32_dr.2_bnno_lr.001_wd.00001_llw5_llbno_co8_si15_seyes_eb0_es0_dreamy_pare'
for data in $(find 'data/' -maxdepth 1 -mindepth 1 -type d -name 'CASP*'); do
    python -m proteins.test \
      "${RUN_PATH}/experiment.latest.yaml" \
      --model state_dict="${RUN_PATH}/model.latest.pt" \
      --test \
        data.input="${data}" \
        data.output="results/allfeatures/$(basename "${data}")" \
        batch_size=200 
done
```

Test GraphQA with residue identity features only:
```bash
RUN_PATH='runs/residueonly_l8_128-512-512_16-64-64_dr.1_bnno_lr.001_wd.00001_llw5_llbno_co8_si15_seyes_eb0_es0_fervent_lichterman'
for data in $(find 'data/' -maxdepth 1 -mindepth 1 -type d -name 'CASP*'); do
    python -m proteins.test \
      "${RUN_PATH}/experiment.latest.yaml" \
      --model state_dict="${RUN_PATH}/model.latest.pt" \
      --test \
        data.input="${data}" \
        data.output="results/residueonly/$(basename "${data}")" \
        batch_size=200 
done
```
