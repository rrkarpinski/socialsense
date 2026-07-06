# Mind the Context

## Dataset

The project uses two datasets, **MANNERSDB+** and **OFFICE-MANNERSDB**, both extensions of the original MANNERSDB.

### Directory structure
```
datasets/
├── MANNERSDBPlus/
│ ├── NAO/
│ ├── Pepper/
│ └── PR2/
└── OFFICE-MANNERSDB/
├── NAO/
├── Pepper/
└── PR2/
```

Each robot directory follows the same structure:
```
[robot]/
├── Annotations/
│ └── *.csv # single CSV, 11 columns (see below)
└── Images/
└── *.png # 1920x1080 images, named by IMAGE_ID
```

**Annotation CSV columns:**

| Column | Description |
|---|---|
| IMAGE_ID | Unique identifier, matches image filename |
| Vacuum Cleaning | Appropriateness score |
| Mopping the Floor | Appropriateness score |
| Carry Warm Food | Appropriateness score |
| Carry Cold Food | Appropriateness score |
| Carry Drinks | Appropriateness score |
| Carry Small Objects | Appropriateness score |
| Carry Large Objects | Appropriateness score |
| Cleaning | Appropriateness score |
| Starting a Conversation | Appropriateness score |
| Reason | Free-text annotator justification |

## Project Structure
```
models/
├── heuristicSplitModel_preprocessing.ipynb # segmentation pipeline: panoptic segmentation → binary masks → social/environmental split
├── heuristicSplitModel.py # model architecture
├── buffers.py # rehearsal buffer for continual learning
└── training_utils.py # shared training helper functions

data_processing/
├── build_data.py # builds (image, label) pairs from raw data
├── data_processing.ipynb # builds HDF5 files from all dataset variants
├── robotfocus.ipynb # builds robot close-up dataset variant
├── data_utils.py # prepares train/validation/test splits
└── raw_dataset_stats.ipynb # MANNERSDB dataset statistics and exploration

experiments/
├── training.ipynb # training entry point
├── evaluation.ipynb # full evaluation pipeline and results tables
├── baseline_*.ipynb # inference and evaluation for each baseline model
├── past_runs.ipynb # log/history of prior training runs
├── corrstats.py # statistical analysis utilities for results
```