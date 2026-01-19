# CropGAN Project

## Overview
A modular, end-to-end DCGAN implementation for generating realistic synthetic images of crop leaves. This project is designed to augment datasets for plant disease classification.

## Features
- **Data Pipeline**: Robust ingestion with `ImageFolder`, resizing, and normalization.
- **DCGAN Model**: Configurable Generator and Discriminator optimized for 64x64 or 128x128 images.
- **Training**: Stable training loop with label smoothing, input noise, and Tensorboard logging.
- **Evaluation**: FID/IS metric placeholders, loss visualization, and downstream classifier benchmarking.
- **Deployment**:
  - **Modules**: Web UI (Streamlit), REST API (FastAPI), CLI.
  - **Containerization**: Docker support.

## Project Structure
```
cropganproject/
├── src/                # Source code
│   ├── dataloader.py   # Data pipeline
│   ├── generator.py    # GAN Generator
│   ├── discriminator.py# GAN Discriminator
│   ├── traindcgan.py   # Training script
│   ├── appleafgan.py   # Streamlit App
│   ├── apileafgan.py   # FastAPI App
│   └── ...
├── configs/            # YAML Configurations
├── checkpoints/        # Saved models
├── samples/            # Generated samples
└── figures/            # Evaluation plots
```

## Quickstart

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Smoke Test (Verification)
Run a complete pipeline test using dummy data:
```bash
# Generate dummy data
python -m src.create_dummy_data

# Run 1-epoch training test
python -m src.traindcgan --config configs/smoketest.yaml
```

### 3. Full Training
```bash
python -m src.traindcgan --config configs/trainconfig.yaml
```

### 4. Generate Images
```bash
python -m src.generate --checkpoint checkpoints/G_epoch_XXX.pth --n 100 --out output/
```

### 5. Deployment
**Web UI**:
```bash
streamlit run src/appleafgan.py
```
**API**:
```bash
uvicorn src.apileafgan:app --reload
```

## UI Tour

The revamped UI ("Field Notebook") offers a specialized workflow for plant pathology research.

### Launching the UI
```bash
streamlit run src/appleafgan.py
```
*Note: If no model checkpoint is found, the UI launches in "Demo Mode" using random noise.*

### Key Sections
1.  **Notebook (Generate)**: Cultivate new synthetic specimens.
    -   Configure Crop, Disease, and Batch Size.
    -   "Attach to Entry" saves your session to `docs/notebook_entries/`.
2.  **Lightbox (Inspect)**: Detailed inspection of generated specimens.
    -   Compare against reference images (placeholder).
    -   Toggle histogram analysis overlays.
3.  **Metrics Bench**: Visualize training progress (FID/IS curves) if `figures/` are available.
4.  **Registry**: View model version history and usage logs.
