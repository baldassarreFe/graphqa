# Protein quality

## Preprocessing
```bash
for f in data/*.h5; do
    rm -r "${f%.*}"
    python -m proteins.dataset preprocess --filepath "$f" --destpath "${f%.*}" 
done
```

## Tensorboard plugins: Custom Scalars Layout and Hyper Parameters
```bash
python -m proteins.layout ./runs
python -m proteins.hparams ./runs
```

## Training
```bash
python -m proteins.train \
    config/train.yaml \
    "tags=['some','tag']" \ 
    --data \
      cutoff=6 \
      separation=yes \
      sigma=8.5 \
      encoding_size=20 \
      encoding_base=10000 \
      residues=yes \
      dssp_features=no \
      self_info=no \
      partial_entropy=no \
    --model \
      config/model.yaml \
    --session \
      config/session.yaml \
      batch_size=100 \
      max_epochs=2
```

## Ablation
```bash
NUM_RUNS_PER_STUDY=5
for f in ./config/ablations/{nodes,edges,layersvscutoff,architecture,localglobalscore}/*.yaml; do
    for i in $(seq ${NUM_RUNS_PER_STUDY}); do
        python -m proteins.train "${f}"
    done
done
```

## Testing
```bash
RUN_PATH=path/to/run/
python -m proteins.test \
  "${RUN_PATH}/experiment.latest.yaml" \
  --model state_dict="${RUN_PATH}/model.latest.pt" \
  --test \
    data.input=data/CASP12 \
    data.output=test/CASP12 \
    batch_size=25
    
for data in $(find './data' -maxdepth 1 -mindepth 1 -type d -name 'CASP*'); do
    python -m proteins.test \
      "${RUN_PATH}/experiment.latest.yaml" \
      --model state_dict="${RUN_PATH}/model.latest.pt" \
      --test \
        data.input="${data}" \
        data.output="test/$(basename "${data}")" \
        batch_size=200 
done
```
