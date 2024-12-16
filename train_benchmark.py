import os
import json
import pickle
import bz2
import random
import shutil
from pathlib import Path

def create_metadata(csv_path):
    """Create a simple metadata file for a CSV."""
    return {
        "table_name": os.path.basename(csv_path),
        "table_description": f"Table from benchmark dataset",
        "dataset_description": "Benchmark dataset for table union search"
    }

def prepare_benchmark_data(benchmark_name, data_root="data", output_root="processed_data", train_ratio=0.7, val_ratio=0.15):
    """
    Prepare benchmark data for TabSketchFM training.
    
    Args:
        benchmark_name: Name of the benchmark (santos, tus, etc.)
        data_root: Root directory containing benchmark data
        output_root: Directory for processed data
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
    """
    # Create necessary directories
    benchmark_dir = os.path.join(data_root, benchmark_name)
    datalake_dir = os.path.join(benchmark_dir, "datalake")
    query_dir = os.path.join(benchmark_dir, "query")
    
    output_dir = os.path.join(output_root, benchmark_name)
    metadata_dir = os.path.join(output_dir, "metadata")
    processed_dir = os.path.join(output_dir, "processed")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Load ground truth
    with open(os.path.join(benchmark_dir, "benchmark.pkl"), 'rb') as f:
        ground_truth = pickle.load(f)
    
    # Collect all unique tables
    all_tables = set()
    for query, matches in ground_truth.items():
        all_tables.add(query)
        all_tables.update(matches)
    
    # Create metadata files and copy CSVs to output directory
    for table in all_tables:
        # Determine source directory (query or datalake)
        if table in ground_truth:
            src_dir = query_dir
        else:
            src_dir = datalake_dir
            
        # Copy CSV file
        src_path = os.path.join(src_dir, table)
        if not os.path.exists(src_path):
            print(f"Warning: {table} not found in {src_dir}")
            continue
            
        # Create metadata
        metadata = create_metadata(table)
        metadata_path = os.path.join(metadata_dir, f"{table}.meta")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    # Create train/val/test splits
    queries = list(ground_truth.keys())
    random.shuffle(queries)
    
    n_queries = len(queries)
    train_size = int(n_queries * train_ratio)
    val_size = int(n_queries * val_ratio)
    
    train_queries = queries[:train_size]
    val_queries = queries[train_size:train_size + val_size]
    test_queries = queries[train_size + val_size:]
    
    # Create splits file
    splits = {
        "train": [],
        "valid": [],
        "test": []
    }
    
    # Helper function to add examples for a query and its matches
    def add_examples(query_list, split):
        for query in query_list:
            # Add query table
            example = {
                "table": os.path.join(query_dir, query),
                "metadata": os.path.join(metadata_dir, f"{query}.meta"),
                "json": os.path.join(processed_dir, f"{hash(query)}.json.bz2"),
                "column": 0  # We'll mask the first column by default
            }
            splits[split].append(example)
            
            # Add matching tables
            for match in ground_truth[query]:
                example = {
                    "table": os.path.join(datalake_dir, match),
                    "metadata": os.path.join(metadata_dir, f"{match}.meta"),
                    "json": os.path.join(processed_dir, f"{hash(match)}.json.bz2"),
                    "column": 0
                }
                splits[split].append(example)
    
    # Create examples for each split
    add_examples(train_queries, "train")
    add_examples(val_queries, "valid")
    add_examples(test_queries, "test")
    
    # Save splits file
    splits_path = os.path.join(output_dir, "splits.json.bz2")
    with bz2.open(splits_path, 'wt') as f:
        json.dump(splits, f, indent=2)
    
    return output_dir, splits_path

def train_tabsketchfm(benchmark_name):
    """Train TabSketchFM on the prepared benchmark data."""
    # Prepare the data
    output_dir, splits_path = prepare_benchmark_data(benchmark_name)
    
    # Create command to run pretrain.py
    cmd = f"""
    python pretrain.py \
        --accelerator {'gpu' if torch.cuda.is_available() else 'cpu'} \
        --devices 1 \
        --max_epochs 40 \
        --save_bert_model \
        --bert_model_path ./models/{benchmark_name}_model \
        --dataset {splits_path} \
        --data_dir {output_dir} \
        --random_seed 0
    """
    
    # Execute the command
    os.system(cmd)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, choices=['santos', 'tus', 'tusLarge', 'pylon'],
                      help='Benchmark to train on')
    args = parser.parse_args()
    
    train_tabsketchfm(args.benchmark)