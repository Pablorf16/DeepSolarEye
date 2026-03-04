# DeepSolarEye - AI Developer Instructions (v3.0)

## Project Overview
**DeepSolarEye** is a machine learning system that predicts solar panel soiling (dirt contamination) from RGB images using a custom PyTorch CNN. It regresses power loss % (0-100%) from panel photos.

**Key Task**: Image (224×224) → CNN → Power Loss % (float)

**Version**: 3.0 (Plan de Ejecución implementado)

## Architecture Overview

### Configuration (src/config.py)
- **Single source of truth** for all hyperparameters
- Key values:
  - `BATCH_SIZE=32`, `LEARNING_RATE=0.001`, `MAX_EPOCHS=50`
  - `ES_PATIENCE=12` (Early Stopping patience)
  - `SCHEDULER_PATIENCE=5`, `SCHEDULER_FACTOR=0.1` (ReduceLROnPlateau)
  - `SEED=42` (reproducibility)
  - `IMG_SIZE=224`
- Categories: `CATEGORY_BINS=[-1, 5, 15, 30, 60, 105]`
- Labels: `['Limpio', 'Leve', 'Moderado', 'Alto', 'Crítico']`

### Data Pipeline (src/data_prep.py)
- **Input**: Raw images in `data/raw/Solar_Panel_Soiling_Image_dataset/PanelImages/`
- **Process**: 
  - Extracts metadata from filenames using regex: `_L_{power_loss}_I_{irradiance}`
  - Stratified 3-way split: 60% train, 20% val, 20% test
  - **Oversampling** on train set only (after split, prevents data leakage)
  - Output: `data/processed/{train_dataset.csv, val_dataset.csv, test_dataset.csv}`
- **Columns**: `filename, date, irradiance, power_loss, dirt_category`

### Model Architecture (src/model.py)
- **Class**: `Net(nn.Module)` - Custom CNN adapted from ImpactNet
- **Key layers**:
  - Initial: `Conv2d(3→16, 7×7)` + AvgPool + Dropout(0.5)
  - 5 Analysis Units (AU1-AU5): Progressive feature extraction with 5×5 convolutions, BatchNorm, ReLU
  - Fully connected: 384→96→96→1 (regression output)
- **v3.0 CRITICAL**: Output is **unbounded** (no sigmoid) for diagnostic purposes
  - Predictions outside [0, 100] indicate learning issues
  - Post-processing clamp is application responsibility

### Dataset & Loading (src/dataset.py)
- **Class**: `SolarPanelDataset(Dataset)` - PyTorch Dataset wrapper
- **Behavior**:
  - Reads CSV with `filename` and `power_loss` columns
  - Images: `Image.open()` → RGB; fallback to black image if missing (logs warning)
  - Transforms: ImageNet normalization (mean/std from config.py)
- **Function**: `get_transforms(phase)` - Returns augmentation pipeline
  - `phase='train'`: RandomHorizontalFlip + RandomVerticalFlip + RandomRotation(180°)
  - `phase='val'/'test'`: Only resize + normalize (deterministic)
  - Raises `ValueError` if phase is invalid

### Training Loop (src/train.py)
- **Entry**: `python -m src.train`
- **Hyperparameters**: Imported from `config.py` (not hardcoded)
- **Features v3.0**:
  - 3-way split: train (oversampleado) / val / test
  - Metrics: RMSE (optimizing) + MAE, R² (diagnostic)
  - Early Stopping: `ES_PATIENCE=12` epochs
  - ReduceLROnPlateau: `patience=5`, `factor=0.1`
  - Checkpoint includes scheduler state_dict
- **Output**: 
  - `saved_models/best_model_v3.pth`
  - `saved_models/checkpoint_v3.pth`
  - `training_log_v3.csv`

### Visualization (src/plot_results.py)
- **Function**: `plot_training_curves_v3(log_file, save_dir)`
- **Output**: `saved_models/training_curves_v3.png`
- Plots: Train RMSE, Val RMSE, Val MAE, Val R², Learning Rate

### EDA (src/eda.py)
- **Entry**: `python -m src.eda`
- **Output**: `reports/figures/`
  - `power_loss_histogram.png`
  - `power_loss_by_category.png`
  - `irradiance_histogram.png`

## Critical Developer Workflows

### 1. Data Preparation
`ash
python -m src.data_prep
# Output: data/processed/{train,val,test}_dataset.csv
`
- Regex pattern: `_L_([0-9\.]+)` and `_I_([0-9\.]+)`
- Stratified split maintains category proportions
- Oversampling balances minority classes in train only

### 2. Training
`ash
python -m src.train
`
- Auto-detects GPU (CUDA) or falls back to CPU
- Saves checkpoint after each epoch (resumable)
- Early stopping monitors val_rmse
- Generates training curves on completion

### 3. EDA & Visualization
`ash
python -m src.eda              # Dataset analysis
python -m src.plot_results     # Training curves
python -m src.test_warp        # Perspective correction test
`

## Project Structure (v3.0)

`
TFG_DeepSolarEye/
├── data/
│   ├── raw/Solar_Panel_Soiling_Image_dataset/PanelImages/
│   └── processed/{train,val,test}_dataset.csv
├── saved_models/
│   ├── best_model_v3.pth
│   ├── checkpoint_v3.pth
│   └── training_curves_v3.png
├── reports/figures/
├── src/
│   ├── __init__.py           # Package exports
│   ├── config.py             # Centralized configuration
│   ├── data_prep.py          # Data extraction + split + oversample
│   ├── dataset.py            # PyTorch Dataset + transforms
│   ├── model.py              # CNN architecture
│   ├── train.py              # Training loop
│   ├── plot_results.py       # Visualization
│   ├── eda.py                # Exploratory analysis
│   └── test_warp.py          # Perspective transform test
├── training_log_v3.csv       # Training metrics history
└── requirements.txt
`

## Dependencies
- **Core ML**: `torch`, `torchvision`
- **Data**: `pandas`, `scikit-learn`, `PIL`, `opencv-python`
- **Viz**: `matplotlib`, `seaborn`
- **Utils**: `tqdm`, `numpy`

## Tips for AI Agents
- **All configs in one place**: Check `src/config.py` before modifying hyperparameters
- **Imports**: Use absolute imports `from src.config import ...`
- **Execution**: Use `python -m src.module` (package-aware)
- **Debugging**: Check `training_log_v3.csv` for training trends
- **Model output**: Unbounded - predictions <0 or >100 indicate issues
