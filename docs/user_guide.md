# CropGAN User Guide

## Setup

- Install dependencies: `pip install -r requirements.txt`
- Create dummy data (optional): `python -m src.create_dummy_data`

## Training

- Smoke test: `python -m src.traindcgan --config configs/smoketest.yaml`
- Full training: `python -m src.traindcgan --config configs/trainconfig.yaml`

## Generate Images (CLI)

- `python -m src.generate --checkpoint checkpoints/G_epoch_10.pth --n 100 --out output/`

## GAN Evaluation

- `python -m src.ganevaluation --checkpoint checkpoints/G_epoch_10.pth --split val --num_images 256`

## Classifier Training & Evaluation

- Baseline: `python -m src.classifiertrain --epochs 5`
- Augmented: `python -m src.classifiertrain --augment_dir Data/Synthetic --epochs 5`
- Evaluate: `python -m src.classifiereval --checkpoint checkpoints/classifier_baseline.pth --split test`

## Web UI

- `streamlit run src/appleafgan.py`

## API

- `uvicorn src.apileafgan:app --reload`

## Notes

- Model registry mapping can be set in configs/trainconfig.yaml under `model_registry`.
- Figures and logs are saved under `figures/` and `logs/`.
