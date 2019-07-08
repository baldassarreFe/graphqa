# Protein quality

## Preprocessing
```bash
python -m proteins.dataset preprocess --filepath data/training_casp9_10.v4.h5 --destpath data/training
python -m proteins.dataset preprocess --filepath data/validation_casp11.v4.h5 --destpath data/validation
python -m proteins.dataset preprocess --filepath data/testing_cameo.v4.h5 --destpath data/testing
```