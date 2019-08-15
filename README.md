# Protein quality

## Preprocessing
```bash
python -m proteins.dataset preprocess --filepath data/training_casp9_10.v4.h5 --destpath data/training
python -m proteins.dataset preprocess --filepath data/validation_casp11.v4.h5 --destpath data/validation
python -m proteins.dataset preprocess --filepath data/testing_cameo.v4.h5 --destpath data/testing
```

## Tensorboard Layout
```bash
python -m proteins.layout ./runs
```

## Training
```bash
python -m proteins.train \
  config/train.yaml "tags=['debug']" \
  --model config/model.yaml \ 
  --session config/session.yaml batch_size=100 max_epochs=2
```