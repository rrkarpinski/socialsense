import argparse
import pandas as pd
from pathlib import Path
import shutil

DATASET_DIR = Path("../../datasets").resolve()
def set_dataset_dir(path_str):
    global DATASET_DIR
    DATASET_DIR = Path(path_str).resolve()
    
PROJECT_DATA_DIR = Path("../../socialsense/data").resolve()
def set_project_data_dir(path_str):
    global PROJECT_DATA_DIR
    PROJECT_DATA_DIR = Path(path_str).resolve()

DATASETS = ["OFFICE-MANNERSDB", "MANNERSDBPlus"]
LABEL_COLS = [
    "Vaccum Cleaning", "Mopping the Floor", "Carry Warm Food",
    "Carry Cold Food", "Carry Drinks", "Carry Small Objects",
    "Carry Large Objects", "Cleaning", "Starting a conversation"
]

def process_csv(csv_path, dataset):
    """Process individual CSV files"""
    df = pd.read_csv(csv_path)
    df = df.drop(columns=df.columns[-1])
    
    # Extract metadata from first column
    first_col = df.columns[0]
    split_data = df[first_col].str.split('_', n=2, expand=True)
    
    df["robot"] = split_data[0]
    df["domain"] = split_data[1]
    df["image_ref"] = split_data[2].astype(int)
    df["dataset"] = dataset

    df = df.drop(columns=[first_col])
    
    return df

def consolidate_data(datasets):
    """Aggregate all CSVs"""
    all_dfs = []
    for dataset in datasets:
        source_path = DATASET_DIR / dataset
        
        for robot in ["NAO", "Pepper", "PR2"]:
            ann_dir = source_path / robot / "Annotations"
            if not ann_dir.exists():
                raise ValueError(f"Labels csv file path ({ann_dir}) doesn't exist")
                
            for csv_file in ann_dir.glob("*.csv"):
                try:
                    df = process_csv(csv_file, dataset)
                    all_dfs.append(df)
                except Exception as e:
                    print(f"Error processing {csv_file}: {str(e)}")
    
    df = pd.concat(all_dfs, ignore_index=True)

    return df

def validate_raw_data(df):
    """Comprehensive data quality checks for raw annotation data"""
    required_columns = {'robot', 'domain', 'image_ref', 'dataset'}

    # Check for any missing columns
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Label value validation (should be between 1 and 5)
    for col in LABEL_COLS:
        if df[col].min() < 1 or df[col].max() > 5:
            raise ValueError(f"Label {col} has invalid range [{df[col].min()}, {df[col].max()}]")

    # Null values check
    null_cols = df.columns[df.isnull().any()].tolist()
    if null_cols:
        raise ValueError(f"Null values found in columns: {null_cols}")

    # Data type and value validation for image_ref
    if not pd.api.types.is_integer_dtype(df['image_ref']):
        raise TypeError("image_ref must be integer type")
    if (df['image_ref'] < 0).any():
        raise ValueError("image_ref contains negative values, which is invalid")

    # Categorical value validation
    valid_robots = {'NAO', 'Pepper', 'PR2'}
    invalid_robots = set(df['robot']) - valid_robots
    if invalid_robots:
        raise ValueError(f"Invalid robot values: {invalid_robots}")

    valid_sources = {'OFFICE-MANNERSDB', 'MANNERSDBPlus'}
    invalid_sources = set(df['dataset']) - valid_sources
    if invalid_sources:
        raise ValueError(f"Invalid source directories: {invalid_sources}")

    return True

def aggregate_labels(df):
    """Aggregate (using mean) multiple annotations per image by image path"""    
    agg_dict = {
        **{col: 'mean' for col in LABEL_COLS},
        **{col: 'first' for col in df.columns.difference(LABEL_COLS).tolist()},
    }
    
    return df.groupby('image_path', as_index=False).agg(agg_dict)

def resolve_image_path(row):
    """Robust path resolution with validation"""
    base_dir = DATASET_DIR / row['dataset'] / row['robot'] / "Images"
    
    if row['dataset'] == "OFFICE-MANNERSDB":
        target = base_dir / f"{row['domain']}_{row['image_ref']}.png"
    else:
        target = next(base_dir.glob(f"{row['image_ref']}_*.png"), None)
    
    if target and target.exists():
        return str(target.resolve())
    return None

def validate_final_data(df):
    """Final validation after aggregation"""
    # Missing image paths
    missing = df[df['image_path'].isnull()]
    if not missing.empty:
        raise FileNotFoundError(
            f"{len(missing)} images missing after aggregation. Examples:\n"
            f"{missing[['robot', 'domain', 'image_ref']].head()}"
        )
    
    # Null values check
    null_cols = df.columns[df.isnull().any()].tolist()
    if null_cols:
        raise ValueError(f"Null values found in columns: {null_cols}")

    # Duplicate image paths
    duplicates = df[df.duplicated('image_path', keep=False)]
    if not duplicates.empty:
        raise RuntimeError(
            f"Duplicate image paths after aggregation:\n"
            f"{duplicates['image_path'].unique()}"
        )

    # Label validity (1-5)
    for col in LABEL_COLS:
        if df[col].min() < 1 or df[col].max() > 5:
            raise ValueError(
                f"Aggregated label {col} out of range: "
                f"[{df[col].min()}, {df[col].max()}]"
            )

    return True


def main(filename, aggregate_data=True):
    try:
        raw_df = consolidate_data(DATASETS)
        validate_raw_data(raw_df)
        raw_df['image_path'] = raw_df.apply(resolve_image_path, axis=1)
        if aggregate_data:
            aggregated_df = aggregate_labels(raw_df)
            validate_final_data(aggregated_df)
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        raise

    print(f"Raw data from datasets processed.")

    data_path = (PROJECT_DATA_DIR / f'{filename}.pkl').as_posix()

    if aggregate_data:
        aggregated_df.to_pickle(data_path)
    else:
        raw_df.to_pickle(data_path)
    
    print(f"Data from datasets saved under {data_path}.")

    return data_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-name", "--data_filename",
        type=str,
        required=True,
        help="Name for the complete dataset file (required).",
    )

    parser.add_argument(
        "-agg", "--aggregate_data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Aggregate labels through mean score per image (default: True). Use --no-aggregate-labels to disable.",
    )
    parser.add_argument(
        "-dir", "--project_data_dir",
        type=str,
        required=True,
        help="Path to the project data directory (required).",
    )

    args = parser.parse_args()

    set_project_data_dir(args.project_data_dir)
    print(f'Project data DIR set to {PROJECT_DATA_DIR}')

    main(
        filename = args.data_filename,
        aggregate_data=args.aggregate_data
    )