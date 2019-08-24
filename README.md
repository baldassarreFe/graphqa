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
python -m proteins.experiment ./runs
```

## Training
```bash
python -m proteins.train \
  config/train.yaml "tags=['debug']" \
  --model config/model.yaml \ 
  --session config/session.yaml batch_size=100 max_epochs=2
```

## Testing
```bash
RUN_PATH=path/to/run/
python -m proteins.test \
  "${RUN_PATH}/experiment.latest.yaml" \
  --model state_dict="${RUN_PATH}/model.latest.yaml" \
  --test \
    data.input=data/CASP12 \
    data.output=test/CASP12 \
    batch_size=25
```
