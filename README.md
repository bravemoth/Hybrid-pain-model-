# Hybrid Pain Model: Deep Learning + Handcrafted Features Fusion

This repository contains the code for a hybrid pain detection system that combines:

1. **Deep model** (`mobilenetv3large_LSTM_model.ipynb`): MobileNetV3 feature extraction with a CNN-LSTM (with attention) classifier
2. **Handcrafted model** (`Action_Units_Random_Forest.ipynb`): OpenFace Action Unit features with a Random Forest classifier
3. **Late fusion** (`Fusion_deep_handcrafted.ipynb`): Probability fusion of deep and handcrafted model outputs (weighted average, RF=60% / Deep=40%)

## Setup

### Environment Variables

All paths are configurable via environment variables. Defaults work on Kaggle. For local use, set these to your actual paths.

| Variable | Used By | Default |
|---|---|---|
| `DATA_DIR` | Deep model | `/kaggle/input/.../modified_pics` |
| `WORK_DIR` | All | `/kaggle/working` |
| `FEATURES_DIR` | Deep model | `{WORK_DIR}/mobilenetv3L_features_data` |
| `MODEL_OUT` | Deep model | `{WORK_DIR}/saved_cnn_lstm.keras` |
| `MOBILENET_W` | Deep model | `/kaggle/input/.../mobilenetv3large_imagenet.weights.h5` |
| `CKPT_PATH` | Deep model | `{WORK_DIR}/ckpt.weights.h5` |
| `FEATURES_PATH` | Deep model | `/kaggle/working/features` |
| `AU_ROOT_DIR` | Random Forest | `/kaggle/input/.../AU_pemf_modified_pics` |
| `SPLIT_CSV` | Random Forest | `/kaggle/input/.../test_sequence_predictions.csv` |
| `DEEP_SRC_PATH` | Fusion | `/kaggle/input/deep/test_sequence_predictions.csv` |
| `RF_SRC_PATH` | Fusion | `/kaggle/input/rf/rf_predictions.csv` |
| `DEEP_CSV_PATH` | Fusion | `{WORK_DIR}/deep_predictions.csv` |
| `RF_CSV_PATH` | Fusion | `{WORK_DIR}/rf_predictions.csv` |
| `MERGED_CSV` | Fusion | `{WORK_DIR}/merged_deep_rf_by_last3.csv` |
| `FUSED_CSV` | Fusion | `{WORK_DIR}/fused_trimmed.csv` |

### Local Installation

```bash
pip install -r requirements.txt
```

## Data Structure

Frame data directory (`DATA_DIR`):
```
DATA_DIR/
├── S001/
│   ├── Neutral/
│   │   └── <sequence>/  (20 frame images)
│   ├── Posed Pain/
│   │   └── <sequence>/
│   └── Algometer Pain/
│       └── <sequence>/
├── S002/
...
```

AU data directory (`AU_ROOT_DIR`):
```
AU_ROOT_DIR/
├── S001/
│   ├── Neutral/
│   │   └── <sequence>/
│   │       └── au_features.csv
├── S002/
...
```

## Execution Order

1. **Train/evaluate deep model**: Run `mobilenetv3large_LSTM_model.ipynb`
   - Outputs: `test_df.csv`, `test_features.npy`, `label_vocab.npy` in `FEATURES_DIR`
   - Final output: `test_sequence_predictions.csv`

2. **Train/evaluate handcrafted model**: Run `Action_Units_Random_Forest.ipynb`
   - Output: `rf_test_results.csv` in `WORK_DIR`

3. **Late fusion**: Run `Fusion_deep_handcrafted.ipynb`
   - Cell 0 copies deep/RF prediction CSVs from `DEEP_SRC_PATH`/`RF_SRC_PATH` to `DEEP_CSV_PATH`/`RF_CSV_PATH`
   - Cell 1 merges them by sequence path
   - Cell 2 computes weighted average (RF 60% / Deep 40%) and outputs metrics + `fused_trimmed.csv`

> **Note**: On local runs, set `DEEP_SRC_PATH` to your `test_sequence_predictions.csv` and `RF_SRC_PATH` to your `rf_test_results.csv`. Or set `DEEP_CSV_PATH`/`RF_CSV_PATH` directly to point to existing CSVs.

## Output Files

| File | Produced By | Description |
|---|---|---|
| `test_sequence_predictions.csv` | Deep model | Per-sequence deep model test predictions |
| `rf_test_results.csv` | Random Forest | Per-sequence RF test predictions |
| `merged_deep_rf_by_last3.csv` | Fusion | Merged deep + RF predictions |
| `fused_trimmed.csv` | Fusion | Final fused predictions (no raw paths) |

## License

MIT License. See `LICENSE` file.
