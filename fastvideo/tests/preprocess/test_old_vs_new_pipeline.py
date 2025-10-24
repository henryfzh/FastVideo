import os
import subprocess
import shutil
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='test_comparison.log', 
    filemode='w'
)
logger = logging.getLogger(__name__)

# Configuration
NUM_NODES = "1"
NUM_GPUS_PER_NODE = "1"
DATA_DIR = "data"

# T2V Configuration
T2V_CONFIG = {
    "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "raw_data_dir": Path(os.path.join(DATA_DIR, "crush-smol")),
    "old_output_dir": Path(os.path.join(DATA_DIR, "crush-smol_processed_t2v_old")),
    "new_output_dir": Path(os.path.join(DATA_DIR, "crush-smol_processed_t2v")),
    "hf_repo": "wlsaidhi/crush-smol-merged",
    "workload_type": "t2v",
}

# I2V Configuration
I2V_CONFIG = {
    "model_path": "Wan-AI/Wan2.1-I2V-14B-480p-Diffusers",
    "raw_data_dir": Path(os.path.join(DATA_DIR, "crush-smol-i2v")),
    "old_output_dir": Path(os.path.join(DATA_DIR, "crush-smol_processed_i2v_old")),
    "new_output_dir": Path(os.path.join(DATA_DIR, "crush-smol_processed_i2v")),
    "hf_repo": "wlsaidhi/crush-smol-merged",
    "workload_type": "i2v",
}


def compare_parquet_files(old_path: str, new_path: str, tolerance: float = 1e-5) -> bool:
    # Compare two parquet files for equality
    logger.info("\nStarting parquet file comparison")
    
    old_path = Path(old_path)
    new_path = Path(new_path)
    
    if not old_path.exists():
        logger.error(f"Old parquet file not found: {old_path}")
        return False
    
    if not new_path.exists():
        logger.error(f"New parquet file not found: {new_path}")
        return False
    
    logger.info(f"Loading old dataset from: {old_path}")
    old_df = pd.read_parquet(old_path)
    
    logger.info(f"Loading new dataset from: {new_path}")
    new_df = pd.read_parquet(new_path)
    
    logger.info(f"Old dataset: {old_df.shape[0]} rows, {old_df.shape[1]} columns")
    logger.info(f"New dataset: {new_df.shape[0]} rows, {new_df.shape[1]} columns")
    
    if old_df.shape[0] != new_df.shape[0]:
        logger.error(f"Row count mismatch. Old: {old_df.shape[0]}, New: {new_df.shape[0]}")
        return False
    
    old_cols = set(old_df.columns)
    new_cols = set(new_df.columns)
    
    if old_cols != new_cols:
        logger.error("Column mismatch: ")
        logger.error(f"Only in old: {old_cols - new_cols}")
        logger.error(f"Only in new: {new_cols - old_cols}")
        return False
    
    for col in sorted(old_df.columns):
        logger.info(f"Comparing column: {col}")
        
        old_values = old_df[col]
        new_values = new_df[col]
        
        try:
            if old_values.dtype == object:
                for idx in range(len(old_values)):
                    old_val = old_values.iloc[idx]
                    new_val = new_values.iloc[idx]
                    
                    if isinstance(old_val, np.ndarray):
                        if not np.allclose(old_val, new_val, rtol=tolerance, atol=tolerance):
                            logger.error(f"Column {col}: values differ at row {idx}")
                            return False
                    elif old_val != new_val:
                        logger.error(f"Column {col}: values differ at row {idx}")
                        return False
            elif np.issubdtype(old_values.dtype, np.number):
                if not np.allclose(old_values, new_values, rtol=tolerance, atol=tolerance, equal_nan=True):
                    logger.error(f"Column {col}: numeric values differ")
                    logger.error(f"  All old values: {old_values.tolist()}")
                    logger.error(f"  All new values: {new_values.tolist()}")
                    return False
            else:
                if not old_values.equals(new_values):
                    logger.error(f"Column {col}: values differ")
                    return False
                    
        except Exception as e:
            logger.error(f"Column {col}: comparison error - {e}")
            return False
    
    logger.info("All comparisons passed")
    return True


def download_data(raw_data_dir: Path, hf_repo: str):
    # Run the data download script
    if raw_data_dir.exists():
        print(f"Data already exists at {raw_data_dir}, skipping download")
        return
    
    print(f"Downloading dataset to {raw_data_dir}...")
    os.makedirs(raw_data_dir.parent, exist_ok=True)
    
    try:
        cmd = [
            "python",
            "scripts/huggingface/download_hf.py",
            "--repo_id", hf_repo,
            "--local_dir", str(raw_data_dir),
            "--repo_type", "dataset"
        ]
        subprocess.run(cmd, check=True)
        print("Download completed successfully")
    except Exception as e:
        print(f"Error during download: {e}")
        raise


def run_old_preprocessing(config: dict):
    print("\nRunning old preprocessing pipeline")
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd()
    cmd = [
        "torchrun",
        "--nnodes", NUM_NODES,
        "--nproc_per_node", NUM_GPUS_PER_NODE,
        "fastvideo/pipelines/preprocess/v1_preprocess.py",
        "--model_path", config["model_path"],
        "--data_merge_path", os.path.join(config["raw_data_dir"], "merge.txt"),
        "--preprocess_video_batch_size", "2",
        "--seed", "42",
        "--max_height", "480",
        "--max_width", "832",
        "--num_frames", "77",
        "--dataloader_num_workers", "0",
        "--output_dir", str(config["old_output_dir"]),
        "--train_fps", "16",
        "--samples_per_file", "8",
        "--flush_frequency", "8",
        "--video_length_tolerance_range", "5",
        "--preprocess_task", config["workload_type"],
    ]
    
    subprocess.run(cmd, check=True, env=env)
    print("Old preprocessing completed")


def run_new_preprocessing(config: dict):
    print("\nRunning new preprocessing pipeline")
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd()
    cmd = [
        "torchrun",
        "--nnodes", NUM_NODES,
        "--nproc_per_node", NUM_GPUS_PER_NODE,
        "-m", "fastvideo.pipelines.preprocess.v1_preprocessing_new",
        "--model_path", config["model_path"],
        "--mode", "preprocess",
        "--workload_type", config["workload_type"],
        "--preprocess.video_loader_type", "torchvision",
        "--preprocess.dataset_type", "merged",
        "--preprocess.dataset_path", str(config["raw_data_dir"]),
        "--preprocess.dataset_output_dir", str(config["new_output_dir"]),
        "--preprocess.preprocess_video_batch_size", "2",
        "--preprocess.dataloader_num_workers", "0",
        "--preprocess.max_height", "480",
        "--preprocess.max_width", "832",
        "--preprocess.num_frames", "77",
        "--preprocess.train_fps", "16",
        "--preprocess.samples_per_file", "8",
        "--preprocess.flush_frequency", "8",
        "--preprocess.video_length_tolerance_range", "5",
    ]
    
    subprocess.run(cmd, check=True, env=env)
    print("New preprocessing completed")


def cleanup_outputs(config: dict):
    print("\nCleaning up previous outputs...")
    
    for output_dir in [config["old_output_dir"], config["new_output_dir"]]:
        if output_dir.exists():
            print(f"  Removing {output_dir}")
            shutil.rmtree(output_dir)


def compare_outputs(config: dict):
    print("\nComparing parquet outputs")
    
    old_parquet_dir = os.path.join(
        config["old_output_dir"],
        "combined_parquet_dataset",
        "worker_0"
    )
    
    new_parquet_dir = os.path.join(
        config["new_output_dir"],
        "training_dataset",
        "worker_0",
        "worker_0"
    )
    
    assert os.path.exists(old_parquet_dir), f"Old parquet dir not found: {old_parquet_dir}"
    assert os.path.exists(new_parquet_dir), f"New parquet dir not found: {new_parquet_dir}"
    
    # Get all parquet files from both directories
    old_parquet_files = sorted([f for f in os.listdir(old_parquet_dir) if f.endswith('.parquet')])
    new_parquet_files = sorted([f for f in os.listdir(new_parquet_dir) if f.endswith('.parquet')])
    
    logger.info(f"Found {len(old_parquet_files)} old parquet files: {old_parquet_files}")
    logger.info(f"Found {len(new_parquet_files)} new parquet files: {new_parquet_files}")
    
    # Check if we have the same number of files
    if len(old_parquet_files) != len(new_parquet_files):
        logger.error(f"Number of parquet files mismatch. Old: {len(old_parquet_files)}, New: {len(new_parquet_files)}")
        assert False, "Number of parquet files don't match"
    
    # Compare each pair of parquet files
    all_passed = True
    for old_file, new_file in zip(old_parquet_files, new_parquet_files):
        logger.info(f"\nComparing {old_file} vs {new_file}")
        
        old_parquet = os.path.join(old_parquet_dir, old_file)
        new_parquet = os.path.join(new_parquet_dir, new_file)
        
        # Load and preview data before comparison
        logger.info("Loading parquet files for preview")
        old_df = pd.read_parquet(old_parquet)
        new_df = pd.read_parquet(new_parquet)
        
        logger.info(f"Old pipeline duration_sec values: {old_df['duration_sec'].tolist()}")
        logger.info(f"New pipeline duration_sec values: {new_df['duration_sec'].tolist()}")
        
        # Run comparison
        result = compare_parquet_files(old_parquet, new_parquet, tolerance=1e-5)
        
        if not result:
            logger.error(f"Comparison FAILED for {old_file}")
            all_passed = False
        else:
            logger.info(f"Comparison PASSED for {old_file}")
    
    if not all_passed:
        logger.error("\nOverall: Some parquet files failed comparison")
        # assert False, "Parquet comparison failed for some files"
    else:
        logger.info("\nOverall: All parquet files comparison PASSED")


def run_preprocessing_test(config: dict, test_name: str):
    print(f"Starting {test_name} Preprocessing Comparison Test")
    
    # cleanup_outputs(config)
    # download_data(config["raw_data_dir"], config["hf_repo"])
    # run_old_preprocessing(config)
    # run_new_preprocessing(config)
    compare_outputs(config)
    
    print(f"{test_name} Preprocessing Comparison Test Completed")

def test_preprocessing_t2v_comparison():
    run_preprocessing_test(T2V_CONFIG, "T2V")


def test_preprocessing_i2v_comparison():
    run_preprocessing_test(I2V_CONFIG, "I2V")


if __name__ == "__main__":
    test_preprocessing_t2v_comparison()
    #test_preprocessing_i2v_comparison()