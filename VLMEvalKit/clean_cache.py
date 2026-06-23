import os
import os.path as osp
import glob
import argparse
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Clean evaluation cache for a specific category to allow partial re-run.")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name, e.g. DermNet_Val_4k or DermNet_Test_1of3")
    parser.add_argument('--category', type=str, default="Reasoning", help="Substring to match category column case-insensitively (default: 'Reasoning')")
    parser.add_argument('--outputs-dir', type=str, default="../outputs", help="Path to outputs directory")
    args = parser.parse_args()

    dataset_name = args.dataset
    category_pattern = args.category.lower()
    outputs_dir = args.outputs_dir

    # 1. Load dataset TSV file to identify indices of the target category
    tsv_path = osp.join('LMUData', f'{dataset_name}.tsv')
    if not osp.exists(tsv_path):
        logger.error(f"Dataset file not found at: {tsv_path}")
        return

    logger.info(f"Loading dataset from {tsv_path}...")
    df_dataset = pd.read_csv(tsv_path, sep='\t')
    
    if 'category' not in df_dataset.columns:
        logger.error(f"Column 'category' not found in dataset. Columns: {list(df_dataset.columns)}")
        return
        
    mask = df_dataset['category'].str.lower().str.contains(category_pattern, na=False)
    target_indices = set(df_dataset[mask]['index'].astype(float).tolist())
    
    logger.info(f"Found {len(target_indices)} questions belonging to category matching '{category_pattern}' out of {len(df_dataset)} total questions.")
    if len(target_indices) == 0:
        logger.warning("No questions found matching the category. Exiting.")
        return

    # 2. Search recursively for prediction files in the outputs directory
    # VLMEvalKit saves predictions as {model_name}_{dataset_name}.xlsx or .tsv
    search_pattern = osp.join(outputs_dir, '**', f'*{dataset_name}.*')
    logger.info(f"Searching for cached prediction files matching pattern: {search_pattern}")
    all_files = glob.glob(search_pattern, recursive=True)
    
    prediction_files = []
    for f in all_files:
        # Filter files that end with dataset_name.xlsx or dataset_name.tsv
        ext = osp.splitext(f)[1].lower()
        base = osp.basename(f)
        if ext in ['.xlsx', '.tsv'] and not ('_score' in base or '_acc' in base or '_eval' in base):
            prediction_files.append(f)
            
    if not prediction_files:
        logger.info("No prediction files found to clean.")
        return

    logger.info(f"Found {len(prediction_files)} prediction file(s) to process:")
    for f in prediction_files:
        logger.info(f"  - {f}")

    # 3. Clean each prediction file and delete corresponding cache/score files
    for pred_file in prediction_files:
        logger.info(f"Processing prediction file: {pred_file}")
        
        # Load the prediction file
        ext = osp.splitext(pred_file)[1].lower()
        try:
            if ext == '.xlsx':
                df_pred = pd.read_excel(pred_file)
            else:
                df_pred = pd.read_csv(pred_file, sep='\t')
        except Exception as e:
            logger.error(f"Failed to read {pred_file}: {e}")
            continue

        if 'index' not in df_pred.columns:
            logger.warning(f"File {pred_file} does not have 'index' column, skipping.")
            continue

        # Count how many target rows exist in this prediction file
        df_pred['index_float'] = pd.to_numeric(df_pred['index'], errors='coerce')
        matching_rows = df_pred[df_pred['index_float'].isin(target_indices)]
        num_to_remove = len(matching_rows)
        
        if num_to_remove == 0:
            logger.info(f"No cached predictions found for target category in {pred_file}. Skipping edits.")
        else:
            # Filter out the matching rows
            df_filtered = df_pred[~df_pred['index_float'].isin(target_indices)]
            df_filtered = df_filtered.drop(columns=['index_float'], errors='ignore')
            
            # Save the file back
            try:
                if ext == '.xlsx':
                    df_filtered.to_excel(pred_file, index=False)
                else:
                    df_filtered.to_csv(pred_file, sep='\t', index=False)
                logger.info(f"Cleaned {num_to_remove} rows from {pred_file}. Saved successfully.")
            except Exception as e:
                logger.error(f"Failed to save cleaned file {pred_file}: {e}")
                continue

        # Delete intermediate checkpoint pkl files and score files in the same directory
        pred_dir = osp.dirname(pred_file)
        model_dir_name = osp.basename(pred_dir) # E.g., deepseek_vl2_int8
        
        # Files to delete:
        # - Any pkl files (checkpoints, PREV.pkl, auxiliary files)
        # - Score files (*_score.xlsx, *_acc.json, etc.)
        files_to_delete = []
        
        # Find score and eval files in the same directory
        for f in glob.glob(osp.join(pred_dir, f'*{dataset_name}*')):
            base = osp.basename(f)
            # Skip the main prediction file we just cleaned
            if f == pred_file:
                continue
            # If it's a score file, evaluation file, or pkl cache file, mark for deletion
            if '_score' in base or '_acc' in base or '_eval' in base or base.endswith('.pkl') or '_checkpoint' in base:
                files_to_delete.append(f)
                
        # Search recursively for chunk pkl files in subdirectories of this model directory
        for f in glob.glob(osp.join(pred_dir, '**', f'*{dataset_name}*.pkl'), recursive=True):
            files_to_delete.append(f)

        # De-duplicate the list of files to delete
        files_to_delete = list(set(files_to_delete))

        if files_to_delete:
            logger.info(f"Deleting intermediate caches and score files for {pred_file}:")
            for f in files_to_delete:
                try:
                    if osp.exists(f):
                        os.remove(f)
                        logger.info(f"  [DELETED] {f}")
                except Exception as e:
                    logger.error(f"  [ERROR] Failed to delete {f}: {e}")
        else:
            logger.info(f"No intermediate score or pkl files found to delete for {pred_file}.")

    logger.info("Cache cleaning completed successfully! You can now run evaluation with --reuse.")

if __name__ == '__main__':
    main()
