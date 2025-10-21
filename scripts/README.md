# Scripts Documentation

This directory contains all the execution scripts for the SSDP project. This document explains how each script works and how they interact with each other.

## 📁 Script Overview

| Script | Purpose | Usage |
|--------|---------|-------|
| `single_run.sh` | Single GPU experiments | Quick testing and development |
| `multi_gpu_run.sh` | Multi-GPU experiments | Production runs and large-scale experiments |
| `hyperparameter_analysis.py` | Parameter optimization | Testing different parameter values |
| `run_hyperparameter_analysis.sh` | Analysis runner | Easy interface for hyperparameter analysis |

## 🖥️ Single GPU Script (`single_run.sh`)

### Purpose
Runs SSDP experiments on a single GPU for quick testing and development.

### How it Works
1. **Sets up environment** - Configures CUDA, working directory
2. **Loads configuration** - Uses hardcoded parameters in the script
3. **Runs experiment** - Executes `main.py` with specified parameters
4. **Saves results** - Outputs to `./results` directory

### Configuration
Edit these variables in the script:
```bash
# Dataset and model configuration
DATASET_NAME=gsm8k                    # Options: gsm8k, math, gsm8ktoy, mathtoy, math100, gsm8k100, gsm8k500
MODEL_NAME=qwen-1.5b                 # Model name
REWARD_MODEL=mistral_prm-7b          # Reward model for evaluation

# Experiment settings
work_dir=./results                    # Output directory
exp_name=test                        # Experiment name

# Config file (modify the --config line in the python command)
python3 main.py \
    --config configs/inference/SSDP.json \  # Change this to your desired config
    --work-dir $work_dir \
    --exp-name $exp_name \
    --data $DATASET_NAME \
    --model $MODEL_NAME \
    --reward_model $REWARD_MODEL \
    --dtype bfloat16 \
    --flash-attn \
    --debug
```

### Usage
```bash
cd scripts
bash single_run.sh
```

## 🚀 Multi-GPU Script (`multi_gpu_run.sh`)

### Purpose
Runs SSDP experiments across multiple GPUs for production-scale experiments.

### How it Works
1. **Detects available GPUs** - Automatically finds and counts GPUs
2. **Partitions dataset** - Splits dataset across available GPUs
3. **Runs parallel experiments** - Each GPU processes a portion of the dataset
4. **Combines results** - Aggregates results from all GPUs
5. **Generates final metrics** - Creates comprehensive output files

### Key Features
- **Automatic GPU detection** - No manual configuration needed
- **Dataset partitioning** - Uses `scripts/partition_dataset.py`
- **Parallel execution** - All GPUs run simultaneously
- **Result combination** - Uses `scripts/combine_results.py`
- **Comprehensive output** - Detailed metrics and timing

### Usage
```bash
# Basic usage (uses defaults)
cd scripts
bash multi_gpu_run.sh

# Custom parameters
bash multi_gpu_run.sh [DATASET_NAME] [MODEL_NAME] [REWARD_MODEL] [EXP_NAME] [WORK_DIR] [CONFIG_FILE]

# Examples
bash multi_gpu_run.sh gsm8k qwen-1.5b mistral_prm-7b my_experiment ./outputs configs/inference/SSDP.json
bash multi_gpu_run.sh math qwen-1.5b mistral_prm-7b math_test ./results configs/inference/DPTS.json
```

### Supported Datasets
- `math` → math_test500_dataset.json
- `gsm8k` → gsm8k_test1319_dataset.json
- `gsm8ktoy` → gsm8k_toy20_dataset.json
- `mathtoy` → math_toy20_dataset.json
- `math100` → math_100_dataset.json
- `gsm8k100` → gsm8k_100_dataset.json
- `gsm8k500` → gsm8k_500_dataset.json

### Output Structure
```
outputs/
├── combined_results/
│   ├── final_evaluation_results.json    # Accuracy metrics
│   ├── combined_inference_metrics.json  # Performance metrics
│   └── combined_clustering_metrics.json # Clustering efficiency
└── gpu_0/, gpu_1/, .../                 # Individual GPU results
```

## 🔬 Hyperparameter Analysis Scripts

### Python Script (`hyperparameter_analysis.py`)

#### Purpose
Tests different parameter values to find optimal settings for SSDP.

#### How it Works
1. **Generates parameter values** - Creates list of values to test
2. **Creates modified configs** - Generates config files for each parameter value
3. **Runs experiments** - Uses `multi_gpu_run.sh` for each parameter value
4. **Collects metrics** - Extracts accuracy and timing data
5. **Exports results** - Saves data to CSV for analysis

#### Configuration Options

**Method 1: Script Configuration (Recommended)**
Edit the `HYPERPARAMETER_CONFIG` dictionary in the script:
```python
HYPERPARAMETER_CONFIG = {
    'parameter_name': 'clustering_threshold',  # Parameter to test
    'min_value': 0.0,                         # Minimum value
    'max_value': 1.0,                         # Maximum value
    'step_size': 0.1,                         # Step size
    'dataset': 'mathtoy',                     # Dataset to use
    'model': 'qwen-1.5b',                    # Model to use
    'reward_model': 'mistral_prm-7b',        # Reward model
    'base_config': 'configs/inference/SSDP.json',  # Base config file
    'work_dir': './hyperparameter_analysis', # Output directory
    'output_csv': None,                       # CSV filename (auto-generated if None)
}
```

**Method 2: Command Line Arguments**
```bash
python3 scripts/hyperparameter_analysis.py \
    --parameter clustering_threshold \
    --min-value 0.0 \
    --max-value 1.0 \
    --step-size 0.1 \
    --dataset mathtoy \
    --model qwen-1.5b
```

#### Supported Parameters
- `clustering_threshold` - Similarity threshold for pruning (0.0-1.0)
- `tree_width` - Maximum tree width (2-8)
- `tree_depth` - Maximum tree depth (8-32)
- `lambda_es` - Early stopping parameter (0.1-1.0)
- `lambda_ds` - Dynamic stopping parameter (0.1-1.0)
- `max_rollout` - Maximum reasoning steps (10-50)

### Bash Script (`run_hyperparameter_analysis.sh`)

#### Purpose
User-friendly interface for running hyperparameter analysis.

#### How it Works
1. **Parses arguments** - Handles command-line parameters
2. **Validates inputs** - Checks required parameters
3. **Calls Python script** - Executes the analysis
4. **Generates visualizations** - Creates plots if visualization script exists
5. **Provides summary** - Shows results and next steps

#### Usage
```bash
# Basic usage
cd scripts
bash run_hyperparameter_analysis.sh --parameter clustering_threshold --min-value 0.0 --max-value 1.0 --step-size 0.1

# Test different parameters
bash run_hyperparameter_analysis.sh --parameter tree_width --min-value 2 --max-value 8 --step-size 1
bash run_hyperparameter_analysis.sh --parameter lambda_es --min-value 0.1 --max-value 1.0 --step-size 0.1

# Quick test with smaller dataset
bash run_hyperparameter_analysis.sh --parameter clustering_threshold --min-value 0.8 --max-value 1.0 --step-size 0.1 --dataset mathtoy
```

## 🔗 Script Dependencies

### Multi-GPU Script Dependencies
The `multi_gpu_run.sh` script relies on several other scripts:

1. **`scripts/partition_dataset.py`** - Splits datasets across GPUs
2. **`scripts/combine_results.py`** - Combines results from all GPUs
3. **`scripts/cleanup_partitions.py`** - Cleans up temporary files

### Hyperparameter Analysis Dependencies
The hyperparameter analysis uses:
1. **`multi_gpu_run.sh`** - For running individual experiments
2. **`scripts/visualize_hyperparameter_results.py`** - For generating plots (optional)

## 📊 Output Files

### Single GPU Output
```
results/
├── config.json                    # Experiment configuration
├── results-*.json                # Detailed results
├── evaluation_results.json       # Accuracy metrics
├── inference_metrics.json        # Performance metrics
└── clustering_metrics.json       # Clustering efficiency
```

### Multi-GPU Output
```
outputs/
├── combined_results/
│   ├── final_evaluation_results.json
│   ├── combined_inference_metrics.json
│   └── combined_clustering_metrics.json
└── gpu_0/, gpu_1/, .../          # Individual GPU results
```

### Hyperparameter Analysis Output
```
hyperparameter_analysis/
├── hyperparameter_analysis_[parameter]_results.csv
├── exp_[parameter]_[value]/       # Individual experiment results
└── config_[parameter]_[value].json  # Modified config files
```

## 🚀 Quick Start Guide

### 1. Single GPU Testing
```bash
# Edit single_run.sh with your parameters
cd scripts
bash single_run.sh
```

### 2. Multi-GPU Production
```bash
# Run with multiple GPUs
cd scripts
bash multi_gpu_run.sh gsm8k qwen-1.5b mistral_prm-7b my_experiment
```

### 3. Hyperparameter Optimization
```bash
# Test clustering threshold
cd scripts
bash run_hyperparameter_analysis.sh --parameter clustering_threshold --min-value 0.0 --max-value 1.0 --step-size 0.1
```

## 🔧 Troubleshooting

### Common Issues

**Script not found errors:**
- Make sure you're in the `scripts/` directory
- Check that all required scripts exist

**GPU detection issues:**
- Ensure CUDA is properly installed
- Check that `nvidia-smi` works

**Dataset not found:**
- Verify dataset files exist in `benchmark/` directory
- Check dataset name spelling

**Permission errors:**
- Make scripts executable: `chmod +x *.sh`
- Check file permissions

### Performance Tips

1. **Use toy datasets** (`mathtoy`, `gsm8ktoy`) for quick testing
2. **Start with single GPU** for development
3. **Use multi-GPU** for production runs
4. **Run hyperparameter analysis** to find optimal parameters

## 📝 Notes

- All scripts should be run from the `scripts/` directory
- The multi-GPU script automatically handles single GPU cases
- Hyperparameter analysis can take a long time - use small datasets for testing
- Results are automatically timestamped to avoid conflicts
- Check the main README.md for overall project setup instructions
