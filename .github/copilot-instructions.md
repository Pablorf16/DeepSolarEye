# DeepSolarEye - AI Developer Instructions

## Project Overview
**DeepSolarEye** is a machine learning system that predicts solar panel soiling (dirt contamination) from RGB images using a custom PyTorch CNN. It regresses power loss % (0-100%) from panel photos.

**Key Task**: Image (224×224) → CNN → Power Loss % (float)

## Architecture Overview

### Data Pipeline (`src/data_prep.py`)
- **Input**: Raw images in `data/raw/Solar_Panel_Soiling_Image_dataset/PanelImages/`
- **Process**: 
  - Extracts metadata from filenames using regex: `_L_{power_loss}_I_{irradiance}`
  - Stratified sampling to balance power loss buckets: [0-5%, 5-15%, 15-30%, 30-60%, 60-100%]
  - Output: CSVs at `data/processed/{train_dataset.csv, test_dataset.csv}` with columns: `filename, date, irradiance, power_loss, dirt_category`
- **Caveat**: Regex parsing is brittle—filenames must follow exact format; returns `None` for invalid names

### Model Architecture (`src/model.py`)
- **Class**: `Net(nn.Module)` - Custom CNN adapted from ImpactNet for regression
- **Key layers**:
  - Initial: `Conv2d(3→16, 7×7)` + AvgPool + Dropout(0.5)
  - 5 Analysis Units (AU1-AU5): Progressive feature extraction with 5×5 convolutions, BatchNorm, ReLU
  - Fully connected: 384→96→96→1 (regression output)
- **Important**: Output is **unbounded**—could predict >100%. Add sigmoid×100 if clamping needed
- **Dropout comment inconsistency**: Code says "384 entradas tras aplanar" but uses 96 (check flattened size)

### Dataset & Loading (`src/dataset.py`)
- **Class**: `SolarPanelDataset(Dataset)` - PyTorch Dataset wrapper
- **Behavior**:
  - Reads CSV with `filename` and `power_loss` columns
  - Images: `Image.open()` → RGB; fallback to black image if missing (silent fail—no error logged)
  - Transforms: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Gotcha**: Missing images don't crash—just return silence, may degrade model quality silently

### Training Loop (`src/train.py`)
- **Entry**: `python src/train.py`
- **Key hyperparameters** (hardcoded):
  - `BATCH_SIZE=32`, `LEARNING_RATE=0.0005`, `MAX_EPOCHS=50`, `PATIENCE=7`
- **Features**:
  - Weighted sampler for class balancing (inverse of category counts)
  - Early stopping on test loss (patience=7 epochs)
  - Checkpoint saving: `saved_models/checkpoint.pth` after each epoch
  - Loss metric: MSE → `torch.nn.MSELoss()`
- **Important**: No validation split—uses test set for early stopping (data leakage risk)
- **Logging**: Creates `training_log.csv` with train/test loss per epoch (file creation logic not visible in snippet)

### Inference (`src/predict.py`)
- **Entry**: `python src/predict.py`
- **Loads**: Model from `saved_models/model_epoch_4.pth` (hardcoded—should be configurable)
- **Process**: Image → Transform → Model → Output power loss %
- **Design**: Expects single image; no batch processing helper

## Critical Developer Workflows

### 1. Data Preparation
```bash
# First time setup: organize images in data/raw/Solar_Panel_Soiling_Image_dataset/PanelImages/
# Filenames MUST match: *_L_{power_loss}_I_{irradiance}*.jpg (e.g., panel_001_L_12.5_I_800_01Jan2021.jpg)
# Then run: python src/data_prep.py
# Output: data/processed/{train_dataset.csv, test_dataset.csv}
```
- **Regex pattern used**: `_L_([0-9\.]+)` and `_I_([0-9\.]+)` and year pattern
- **Validation**: Power loss clamped to [0, 100]; rows with invalid names silently dropped
- **Risk**: If regex changes, rerun data_prep; stale CSVs cause silent model degradation

### 2. Training
```bash
python src/train.py
```
- Runs on GPU if available (`DEVICE='cuda' if torch.cuda.is_available() else 'cpu'`)
- Saves checkpoints: `saved_models/checkpoint.pth` (overwritten each epoch)
- Saves best model by epoch: `saved_models/model_epoch_{i}.pth` (5 epochs saved here)
- **Important**: Training log location not exposed in code snippet

### 3. Inference
```bash
python src/predict.py
```
- Hardcoded paths in `predict.py`—modify `MODEL_PATH` and `TEST_IMAGE_PATH` variables
- No error if image not found; just prints error and exits

### 4. Notebook (`main.ipynb`)
- Contains full workflow (data load → preprocess → train → evaluate)
- Currently not executed; check cell outputs if you run it

## Project-Specific Patterns & Conventions

### Filename Extraction
- All filenames encode metadata: `{name}_L_{loss}_I_{irrad}_{date}.jpg`
- Extraction done once in `data_prep.py`; subsequent steps use CSVs
- If you modify filenames, rerun `data_prep.py`

### Model Configuration
- Hyperparameters hardcoded in `train.py` (not config file)
- Model architecture in `model.py` is fixed (ImpactNet port)
- No model versioning—just epoch numbers

### Device Management
- All files check `torch.cuda.is_available()`; gracefully fallback to CPU
- No device configuration file; hardcoded per script

### Path Resolution
- Uses `os.path` with `__file__` to derive `BASE_DIR`
- Works from any working directory (relative paths computed dynamically)

## Integration Points & Dependencies

### External Dependencies
- **Core ML**: `torch`, `torchvision`, `torch.nn`
- **Data**: `pandas`, `scikit-learn` (train_test_split), `PIL`, `opencv-python`
- **Viz**: `matplotlib`
- **Utils**: `tqdm`, `numpy`
- **Missing**: No scipy explicitly listed but might be used

### Cross-Component Communication
1. **data_prep.py** → CSVs → **dataset.py** → **train.py**/inference
2. **train.py** → Model weights → `saved_models/*.pth` → **predict.py**
3. **train.py** → Metrics → `training_log.csv` → **plot_results.py**

### Data Flow
```
Raw Images 
  ↓ (data_prep.py: regex extraction)
train/test CSVs
  ↓ (dataset.py: PyTorch Dataset)
DataLoader (train.py)
  ↓ (training loop)
Model checkpoints (.pth)
  ↓ (predict.py: loads + infers)
Power Loss predictions
```

## Known Issues & Workarounds

| Issue | Impact | Workaround |
|-------|--------|-----------|
| Missing images silently handled | Model trains on blanks, degraded performance | Validate filenames match regex; check CSV file counts |
| No validation set | Early stopping uses test set (leakage) | Manually split test set 80/20 for validation |
| Model output unbounded | Predictions potentially >100% or <0% | Add sigmoid or clamp in predict.py |
| Hardcoded model path in predict.py | Must edit code to switch models | Use argparse or env vars |
| No error logging | Failures silent; debug via print statements | Wrap key functions with try/except + logging |
| Inconsistent tensor shapes | Index mismatch in FC layers | Verify flattened size matches 384 or update to 96 |

## Improvement Priorities
1. ✅ Add logging (Python `logging` module)
2. ✅ Fix model output clamping + add evaluation metrics (MAE, RMSE, R²)
3. ✅ Add config file for hyperparameters (JSON or YAML)
4. ✅ Implement proper validation split (3-way: train/val/test)
5. ✅ Add seed setting for reproducibility
6. ⚠️ Refactor hardcoded paths to argparse/config
7. ⚠️ Unit tests for data_prep regex and model forward pass

## Commands Reference
```bash
# Setup
pip install -r requirements.txt

# Run pipeline
python src/data_prep.py          # Extract metadata from filenames
python src/train.py               # Train model
python src/predict.py             # Predict on single image

# Utilities
python src/eda.py                 # Exploratory data analysis
python src/plot_results.py         # Plot training curves
python src/test_warp.py            # Test image warping utils

# Notebook
jupyter notebook main.ipynb        # Full interactive workflow
```

## Tips for AI Agents
- **Before modifying data_prep.py**: Check sample filenames in `data/raw/...` to verify regex pattern
- **Before retraining**: Backup `saved_models/` and review `training_log.csv` for trends
- **When adding features**: Keep dataset.csv schema consistent; migration is manual
- **Testing predictions**: Use images from `data/raw/` that you know the expected loss for
- **Debugging model inference**: Check input tensor shape is [1, 3, 224, 224] and model.eval() is called
