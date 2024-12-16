import os
import json
import pickle
import bz2
import random
import pandas as pd
from pathlib import Path
from tabsketchfm.data_processing.data_prep import prep_data
import shutil

def prepare_benchmark_data(benchmark_name):
    """Prepare data for training TabSketchFM on a benchmark dataset."""
    
    # Create directory structure
    output_dir = Path("processed_data") / benchmark_name
    processed_dir = output_dir / "processed"
    metadata_dir = output_dir / "metadata"
    
    # Clean previous runs if they exist
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    # Load ground truth
    data_dir = Path("data") / benchmark_name
    if not data_dir.exists():
        raise ValueError(f"Benchmark data directory not found: {data_dir}")

    # Process all tables from both query and datalake directories
    processed_files = {}
    for subdir in ['query', 'datalake']:
        dir_path = data_dir / subdir
        if not dir_path.exists():
            continue
            
        # Process each CSV file
        for csv_file in dir_path.glob("*.csv"):
            # Create metadata for the table
            metadata = {
                "table_name": csv_file.name,
                "table_description": f"Table from {benchmark_name} {subdir}",
                "dataset_description": f"{benchmark_name} benchmark dataset"
            }
            
            # Save metadata
            metadata_file = metadata_dir / f"{csv_file.name}.meta"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
            
            # Process table
            prep_data(str(csv_file), str(processed_dir), str(metadata_file))
            
            # Get the processed json file name (will be hash of original filename)
            json_files = list(processed_dir.glob(f"*.json.bz2"))
            if json_files:  # Take the most recent one
                processed_files[csv_file.name] = f"processed/{json_files[-1].name}"  # Include processed/ in path

    # Create train/valid/test splits
    tables = list(processed_files.keys())
    random.shuffle(tables)
    
    n = len(tables)
    train_idx = int(0.7 * n)
    val_idx = int(0.85 * n)
    
    splits = {
        "train": [],
        "valid": [],
        "test": []
    }
    
    # Helper function to add tables to splits
    def add_to_split(table_name, split):
        if table_name not in processed_files:
            return
            
        # For each table, add entries for each column
        try:
            df = pd.read_csv(data_dir / 'query' / table_name if (data_dir / 'query' / table_name).exists() 
                           else data_dir / 'datalake' / table_name)
            num_cols = len(df.columns)
            
            for col_idx in range(num_cols):
                splits[split].append({
                    "table": table_name,
                    "json": processed_files[table_name],  # Now includes processed/ in path
                    "column": col_idx
                })
        except Exception as e:
            print(f"Error processing {table_name}: {str(e)}")
    
    # Add tables to splits
    for table in tables[:train_idx]:
        add_to_split(table, "train")
    for table in tables[train_idx:val_idx]:
        add_to_split(table, "valid")
    for table in tables[val_idx:]:
        add_to_split(table, "test")

    # Save splits file
    splits_path = output_dir / "splits.json.bz2"
    with bz2.open(splits_path, 'wt') as f:
        json.dump(splits, f)

    print(f"Created {len(splits['train'])} training examples")
    print(f"Created {len(splits['valid'])} validation examples")
    print(f"Created {len(splits['test'])} test examples")

    return str(output_dir), str(splits_path)

def train_tabsketchfm(benchmark_name):
    """Train TabSketchFM on a benchmark dataset."""
    output_dir, splits_path = prepare_benchmark_data(benchmark_name)
    
    cmd = f"""
    python pretrain.py \\
        --accelerator 'gpu' \\
        --devices 1 \\
        --max_epochs 40 \\
        --save_bert_model \\
        --bert_model_path ./models/{benchmark_name}_model \\
        --dataset {splits_path} \\
        --data_dir {output_dir} \\
        --random_seed 0
    """
    os.system(cmd)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, required=True,
                       choices=['santos', 'tus', 'tusLarge', 'pylon'])
    args = parser.parse_args()
    train_tabsketchfm(args.benchmark)