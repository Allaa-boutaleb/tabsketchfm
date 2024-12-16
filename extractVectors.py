import os
import pickle
import torch
from pathlib import Path
import bz2
import json
from loguru import logger
from transformers import AutoConfig, AutoTokenizer
from tabsketchfm import TableSimilarityTokenizer
from tabsketchfm.models.tabsketchfm_finetune import FinetuneTabSketchFM

def extract_embeddings(model_path, data_dir):
    """Extract embeddings for all tables using trained TabSketchFM model."""
    vectors_dir = Path("vectors/santos")
    vectors_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer
    config = AutoConfig.from_pretrained(model_path)
    config.task_specific_params = {'hash_input_size': config.hidden_size}
    tokenizer_base = AutoTokenizer.from_pretrained(model_path)
    tokenizer = TableSimilarityTokenizer(tokenizer=tokenizer_base, config=config)
    
    model = FinetuneTabSketchFM(
        model_name_or_path=model_path,
        config=config,
        learning_rate=2e-5,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8
    )
    model.eval()
    
    # Initialize storage for datalake and query embeddings
    datalake_embeddings = []
    query_embeddings = []
    
    # Process all files in processed directory
    processed_dir = Path(data_dir) / "processed"
    
    for json_file in processed_dir.glob("*.json.bz2"):
        with bz2.open(json_file, 'rt') as f:
            data = json.load(f)
            filepath = Path(data['table_metadata']['file_name'])
            subdir = filepath.parent.name  # Should be either 'query' or 'datalake'
            filename = filepath.name
            
            logger.info(f"Processing {filename}")
            try:
                with torch.no_grad():
                    inputs = tokenizer.tokenize_function(data)
                    for key in inputs:
                        if torch.is_tensor(inputs[key]) and inputs[key].dim() == 1:
                            inputs[key] = inputs[key].unsqueeze(0)
                    
                    outputs = model.model(**inputs)
                    embeddings = outputs.pooler_output.cpu().numpy()
                    
                    if subdir == 'datalake':
                        datalake_embeddings.append((filename, embeddings))
                    elif subdir == 'query':
                        query_embeddings.append((filename, embeddings))
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                continue
    
    # Save embeddings
    with open(vectors_dir / "datalake_vectors.pkl", 'wb') as f:
        pickle.dump(datalake_embeddings, f)
    with open(vectors_dir / "query_vectors.pkl", 'wb') as f:
        pickle.dump(query_embeddings, f)
    
    logger.info(f"Saved {len(datalake_embeddings)} datalake embeddings")
    logger.info(f"Saved {len(query_embeddings)} query embeddings")

if __name__ == "__main__":
    import argparse
    
    logger.remove()
    logger.add("extractVectors.log", rotation="500 MB")
    logger.add(lambda msg: print(msg), level="INFO")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained TabSketchFM model')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to processed data directory (processed_data/santos)')
    
    args = parser.parse_args()
    extract_embeddings(args.model_path, args.data_dir)