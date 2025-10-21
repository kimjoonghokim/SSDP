#!/bin/bash
"""
Hyperparameter Analysis Runner Script
====================================

This script runs hyperparameter analysis for any parameter in the config file.

Usage:
    bash scripts/run_hyperparameter_analysis.sh [options]

Examples:
    # Test clustering threshold
    bash scripts/run_hyperparameter_analysis.sh --parameter clustering_threshold --min-value 0.0 --max-value 1.0 --step-size 0.1

    # Test tree_width
    bash scripts/run_hyperparameter_analysis.sh --parameter tree_width --min-value 2 --max-value 8 --step-size 1

    # Test lambda_es
    bash scripts/run_hyperparameter_analysis.sh --parameter lambda_es --min-value 0.1 --max-value 1.0 --step-size 0.1

    # Quick test with smaller dataset
    bash scripts/run_hyperparameter_analysis.sh --parameter clustering_threshold --min-value 0.8 --max-value 1.0 --step-size 0.1 --dataset mathtoy
"""

set -e  # Exit on any error

# Default parameters
PARAMETER=""
MIN_VALUE=""
MAX_VALUE=""
STEP_SIZE=""
DATASET="mathtoy"
MODEL="qwen-1.5b"
REWARD_MODEL="mistral_prm-7b"
BASE_CONFIG="configs/inference/SSDP.json"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --parameter)
            PARAMETER="$2"
            shift 2
            ;;
        --min-value)
            MIN_VALUE="$2"
            shift 2
            ;;
        --max-value)
            MAX_VALUE="$2"
            shift 2
            ;;
        --step-size)
            STEP_SIZE="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --base-config)
            BASE_CONFIG="$2"
            shift 2
            ;;
        --help)
            echo "Hyperparameter Analysis Runner Script"
            echo "====================================="
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Required Options:"
            echo "  --parameter STRING      Parameter name to test (e.g., clustering_threshold, tree_width)"
            echo "  --min-value FLOAT       Minimum parameter value"
            echo "  --max-value FLOAT       Maximum parameter value"
            echo "  --step-size FLOAT       Step size for parameter values"
            echo ""
            echo "Optional Options:"
            echo "  --dataset STRING        Dataset to use (default: mathtoy)"
            echo "  --model STRING          Model to use (default: qwen-1.5b)"
            echo "  --base-config STRING    Base config file (default: configs/inference/SSDP.json)"
            echo "  --help                  Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --parameter clustering_threshold --min-value 0.0 --max-value 1.0 --step-size 0.1"
            echo "  $0 --parameter tree_width --min-value 2 --max-value 8 --step-size 1"
            echo "  $0 --parameter lambda_es --min-value 0.1 --max-value 1.0 --step-size 0.1"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$PARAMETER" ]; then
    echo "Error: --parameter is required"
    exit 1
fi

if [ -z "$MIN_VALUE" ]; then
    echo "Error: --min-value is required"
    exit 1
fi

if [ -z "$MAX_VALUE" ]; then
    echo "Error: --max-value is required"
    exit 1
fi

if [ -z "$STEP_SIZE" ]; then
    echo "Error: --step-size is required"
    exit 1
fi

# Calculate number of experiments
NUM_EXPERIMENTS=$(python3 -c "
min_val = $MIN_VALUE
max_val = $MAX_VALUE
step = $STEP_SIZE
count = 0
current = min_val
while current <= max_val:
    count += 1
    current = round(current + step, 3)
print(count)
")

echo "=========================================="
echo "Hyperparameter Analysis Configuration"
echo "=========================================="
echo "Parameter: $PARAMETER"
echo "Value range: $MIN_VALUE to $MAX_VALUE (step: $STEP_SIZE)"
echo "Number of experiments: $NUM_EXPERIMENTS"
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Base config: $BASE_CONFIG"
echo "=========================================="

# Check if required files exist
if [ ! -f "$BASE_CONFIG" ]; then
    echo "Error: Base config file not found: $BASE_CONFIG"
    exit 1
fi

if [ ! -f "scripts/hyperparameter_analysis.py" ]; then
    echo "Error: hyperparameter_analysis.py script not found"
    exit 1
fi

# Create work directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
WORK_DIR="./hyperparameter_analysis_${PARAMETER}_${TIMESTAMP}"
echo "Work directory: $WORK_DIR"

# Run hyperparameter analysis
echo "Starting hyperparameter analysis..."
python3 scripts/hyperparameter_analysis.py \
    --base-config "$BASE_CONFIG" \
    --parameter "$PARAMETER" \
    --min-value "$MIN_VALUE" \
    --max-value "$MAX_VALUE" \
    --step-size "$STEP_SIZE" \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --reward-model "$REWARD_MODEL" \
    --work-dir "$WORK_DIR"

# Check if analysis completed successfully
if [ $? -ne 0 ]; then
    echo "Hyperparameter analysis failed!"
    exit 1
fi

# Find the CSV results file
CSV_FILE="$WORK_DIR/hyperparameter_analysis_${PARAMETER}_results.csv"
if [ ! -f "$CSV_FILE" ]; then
    echo "Error: Results CSV file not found: $CSV_FILE"
    exit 1
fi

echo "Analysis completed! Results saved to: $CSV_FILE"

# Check if visualization script exists and run it
if [ -f "scripts/visualize_hyperparameter_results.py" ]; then
    echo "Generating visualizations..."
    python3 scripts/visualize_hyperparameter_results.py "$CSV_FILE" --output-dir "$WORK_DIR"
    
    if [ $? -eq 0 ]; then
        echo "Visualizations saved to: $WORK_DIR"
        echo "  - hyperparameter_analysis_overview.png"
        echo "  - hyperparameter_analysis_detailed.png"
    else
        echo "Warning: Visualization generation failed"
    fi
else
    echo "Warning: Visualization script not found. You can create plots manually using the CSV file."
fi

echo ""
echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
echo "Results directory: $WORK_DIR"
echo "CSV file: $CSV_FILE"
echo ""
echo "You can now:"
echo "1. View the CSV file to see raw data"
echo "2. Open the PNG files to see visualizations"
echo "3. Use the data to create custom graphs"
echo "4. Analyze the relationship between $PARAMETER, time, and accuracy"
echo "=========================================="
