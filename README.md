# Semantic Similarity Based Dynamic Pruning (SSDP) for Tree-of-Thought Reasoning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-red.svg)](https://pytorch.org/)

This repository contains the official implementation of **Semantic Similarity Based Dynamic Pruning (SSDP)** for Tree-of-Thought reasoning. SSDP is an advanced method that improves the efficiency of large language model reasoning by dynamically pruning semantically similar nodes in the reasoning tree, reducing computational overhead while maintaining reasoning quality.

## 🔗 **Related Work & Attribution**

This implementation is built on top of the [**Dynamic Parallel Tree Search (DPTS)**](https://arxiv.org/abs/2502.16235) framework with significant enhancements for semantic similarity-based pruning.

**Original DPTS Paper:**
> Dynamic Parallel Tree Search for Efficient LLM Reasoning  
> Authors: Ding, Yifu and Jiang, Wentao and Liu, Shunyu and Jing, Yongcheng and Guo, Jinyang and Wang, Yingjie and Zhang, Jing and Wang, Zengmao and Liu, Ziwei and Du, Bo and Liu, Xianglong and Tao, Dacheng  
> arXiv: [2502.16235](https://arxiv.org/abs/2502.16235)

**Our SSDP Paper:**
📄 **Paper**: [Chopping Trees: Semantic Similarity Based Dynamic Pruning for Tree-of-Thought Reasoning](https://example.com/paper-link)

## 🚀 Key Features

- **Semantic Clustering**: Uses embedding models to identify and cluster semantically similar reasoning paths
- **Dynamic Pruning**: Intelligently prunes redundant nodes based on similarity thresholds
- **Multi-GPU Support**: Efficient distributed inference across multiple GPUs
- **Flexible Configuration**: Easy-to-use JSON configuration system
- **Multiple Datasets**: Support for GSM8K, MATH, and other reasoning benchmarks

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Dataset Support](#dataset-support)
- [Citation](#citation)
- [Troubleshooting](#troubleshooting)

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 16GB+ RAM (for large models)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/SSDP.git
   cd SSDP
   ```

2. **Run the setup script:**
   ```bash
   bash setup.sh
   ```

   This will:
   - Upgrade pip and install build tools
   - Install PyTorch with CUDA support
   - Install all required dependencies

3. **Verify installation:**
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## 🚀 Quick Start

### Single GPU Inference

```bash
# Navigate to scripts directory and run single GPU script
cd scripts
bash single_run.sh
```

### Multi-GPU Inference

```bash
# Navigate to scripts directory and run multi-GPU script
cd scripts
bash multi_gpu_run.sh
```

### Basic Example

```bash
# Example with GSM8K dataset using single GPU script
cd scripts
bash single_run.sh
```

## ⚙️ Configuration

SSDP uses JSON configuration files to control behavior. Key parameters include:

### Core SSDP Parameters

```json
{
    "config": {
        "enable_clustering": true,
        "clustering_threshold": 0.75,
        "clustering_method": "cosine_similarity",
        "embedding_model": "sentence-transformers",
        "tree_width": 4,
        "tree_depth": 16,
        "max_rollout": 20
    }
}
```

### Available Configuration Files

- `configs/inference/SSDP.json` - Full SSDP configuration with clustering
- `configs/inference/DPTS.json` - Base DPTS configuration without clustering

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `enable_clustering` | Enable semantic clustering | `true` |
| `clustering_threshold` | Similarity threshold for pruning (0.0-1.0) | `0.75` |
| `clustering_method` | Clustering algorithm | `"cosine_similarity"` |
| `tree_width` | Maximum tree width | `4` |
| `tree_depth` | Maximum tree depth | `16` |
| `max_rollout` | Maximum reasoning steps | `20` |

## 📊 Usage Examples

### Configuring the Scripts

Before running experiments, you need to configure the model and dataset parameters in the bash scripts:

#### Single GPU Script (`single_run.sh`)

Edit the following variables in `single_run.sh`:

```bash
# Dataset configuration
DATASET_NAME=gsm8k                    # Options: gsm8k, math, gsm8ktoy, mathtoy, math100, gsm8k100, gsm8k500
MODEL_NAME=qwen-1.5b                  # Model name (e.g., qwen-1.5b, qwen-7b, etc.)
REWARD_MODEL=mistral_prm-7b          # Reward model for evaluation

# Experiment configuration
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

#### Multi-GPU Script (`multi_gpu_run.sh`)

The multi-GPU script accepts command-line arguments:

```bash
# Usage: bash multi_gpu_run.sh [DATASET_NAME] [MODEL_NAME] [REWARD_MODEL] [EXP_NAME] [WORK_DIR] [CONFIG_FILE]

# Example with custom parameters
bash multi_gpu_run.sh gsm8k qwen-1.5b mistral_prm-7b my_experiment ./outputs configs/inference/SSDP.json
```

### Running on Different Datasets

```bash
# GSM8K mathematical reasoning (single GPU)
# Edit single_run.sh: DATASET_NAME=gsm8k
cd scripts
bash single_run.sh

# MATH competition problems (multi-GPU)
cd scripts
bash multi_gpu_run.sh math qwen-1.5b mistral_prm-7b math_experiment

# Small toy datasets for testing
cd scripts
bash multi_gpu_run.sh gsm8ktoy qwen-1.5b mistral_prm-7b toy_test
```

### Custom Configuration Files

Create your own configuration file based on the existing ones:

#### For Single GPU Script:
```bash
# Copy and modify existing config
cp configs/inference/SSDP.json configs/inference/my_config.json

# Edit my_config.json with your parameters
# Then edit single_run.sh and change the --config line:
# --config configs/inference/my_config.json \

cd scripts
bash single_run.sh
```

#### For Multi-GPU Script:
```bash
# Copy and modify existing config
cp configs/inference/SSDP.json configs/inference/my_config.json

# Edit my_config.json with your parameters
# Then use it with multi-GPU script
cd scripts
bash multi_gpu_run.sh gsm8k qwen-1.5b mistral_prm-7b my_experiment ./outputs configs/inference/my_config.json
```

## 📁 Dataset Support

SSDP supports multiple reasoning datasets:

- **GSM8K**: Grade school math problems
- **MATH**: Competition-level mathematics
- **Custom**: User-defined datasets

### Dataset Configuration

Datasets are automatically loaded based on the `--data` parameter. Each dataset includes:
- Problem statements
- Ground truth solutions
- Evaluation metrics

## 📈 Results and Output

SSDP generates comprehensive output including:

- **Results**: Detailed reasoning paths and final answers
- **Metrics**: Accuracy, efficiency, and clustering statistics
- **Logs**: Detailed execution logs and performance metrics
- **Configurations**: Complete experiment configuration

### Output Structure

```
outputs/
├── config.json              # Experiment configuration
├── results-*.json           # Detailed results
├── evaluation_results.json  # Accuracy metrics
├── inference_metrics.json   # Performance metrics
└── clustering_metrics.json  # Clustering efficiency
```

## 🔬 Citation

If you use SSDP in your research, please cite both our paper and the original DPTS work:

### Our SSDP Paper:
```bibtex
@article{ssdp2025,
  title={Chopping Trees: Semantic Similarity Based Dynamic Pruning for Tree-of-Thought Reasoning},
  author={Joongho Kim, Xirui Huang, Zarreen Reza, Gabriel Grand, Kevin Zhu, Ryan Lagasse},
  journal={NeurIPS 2025 Workshop on Efficient Reasoning},
  year={2025}
}
```

### Original DPTS Paper (Please also cite):
```bibtex
@article{ding2025dynamic,
  title={Dynamic parallel tree search for efficient llm reasoning},
  author={Ding, Yifu and Jiang, Wentao and Liu, Shunyu and Jing, Yongcheng and Guo, Jinyang and Wang, Yingjie and Zhang, Jing and Wang, Zengmao and Liu, Ziwei and Du, Bo and Liu, Xianglong and Tao, Dacheng},
  journal={arXiv preprint arXiv:2502.16235},
  year={2025}
}
```

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory:**
```bash
# Reduce batch size or model size
# Try using a smaller model or reducing tree_width/tree_depth in config
```

**Installation Issues:**
```bash
# Clean installation
pip uninstall torch torchaudio torchvision -y
pip install torch==2.1.2 torchaudio==2.1.2 torchvision==0.16.2
pip install -r requirements.txt
```

**Model Loading Issues:**
- Ensure model paths are correct
- Check model compatibility with your hardware
- Verify sufficient disk space for model weights

### Performance Optimization

- Adjust `clustering_threshold` to balance efficiency and accuracy
- Use multiple GPUs with `multi_gpu_run.sh` for faster processing
- Reduce `tree_width` and `tree_depth` in config for faster experimentation

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

The Apache License 2.0 provides:
- ✅ **Permissive licensing** - Allows commercial and non-commercial use
- ✅ **Patent protection** - Explicit patent grant from contributors
- ✅ **Attribution required** - Must include copyright notice
- ✅ **Modification allowed** - Can create derivative works
- ✅ **Distribution allowed** - Can redistribute with or without changes
  
---

## 🙏 **Acknowledgments**

We gratefully acknowledge the original **Dynamic Parallel Tree Search (DPTS)** team for their foundational work. This implementation extends their framework with semantic similarity-based pruning capabilities. Please refer to the original DPTS paper for the theoretical foundations and baseline implementation.

**Original DPTS Repository**: [https://github.com/yifu-ding/DPTS](https://github.com/yifu-ding/DPTS)  
**Original DPTS Paper**: [Dynamic Parallel Tree Search for Efficient LLM Reasoning](https://arxiv.org/abs/2502.16235)
