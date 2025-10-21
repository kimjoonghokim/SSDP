#!/bin/bash

# Multi-GPU Experiment Runner
# This script partitions datasets across available GPUs and runs experiments in parallel
# without using accelerate, using our custom partitioning and combination scripts
# 
# Supported datasets (from data_config.py):
#   math     -> math_test500_dataset.json
#   gsm8k    -> gsm8k_test1319_dataset.json  
#   gsm8ktoy -> gsm8k_toy20_dataset.json
#   mathtoy  -> math_toy20_dataset.json

set -e  # Exit on any error

# Configuration
# Only change workspace if not already in the project root
if [ ! -d "./benchmark" ]; then
    WORKSPACE=../
    cd $WORKSPACE
fi
export BENCHMARK_ROOT='./benchmark'
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Default parameters (can be overridden by command line arguments)
DATASET_NAME=${1:-mathtoy}
MODEL_NAME=${2:-qwen-1.5b}
REWARD_MODEL=${3:-mistral_prm-7b}
EXP_NAME=${4:-multi_gpu_test}
WORK_DIR=${5:-./outputs}
CONFIG_FILE=${6:-configs/inference/DPTS.json}

# Detect available GPUs
detect_gpus() {
    if command -v nvidia-smi &> /dev/null; then
        # Use nvidia-smi to detect GPUs
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        echo "Detected $GPU_COUNT NVIDIA GPUs"
    elif command -v lspci &> /dev/null; then
        # Fallback to lspci for other GPU types
        GPU_COUNT=$(lspci | grep -i vga | grep -i nvidia | wc -l)
        echo "Detected $GPU_COUNT GPUs via lspci"
    else
        echo "Warning: Could not detect GPUs. Using single GPU mode."
        GPU_COUNT=1
    fi
    
    if [ $GPU_COUNT -eq 0 ]; then
        echo "Error: No GPUs detected!"
        exit 1
    fi
    
    echo "Using $GPU_COUNT GPUs for parallel execution"
}

# Create experiment directories
setup_experiment_dirs() {
    echo "Setting up experiment directories..."
    
    # Create main work directory
    mkdir -p $WORK_DIR
    
    # Create subdirectories for each GPU
    for i in $(seq 0 $((GPU_COUNT-1))); do
        mkdir -p $WORK_DIR/gpu_${i}
    done
    
    # Create temp directory for partitions
    mkdir -p $BENCHMARK_ROOT/temp_partitions
    
    echo "Created experiment directories for $GPU_COUNT GPUs"
}

# Partition dataset
partition_dataset() {
    echo "Partitioning dataset: $DATASET_NAME"
    
    # Map dataset names to their actual file paths (matching data_config.py)
    case $DATASET_NAME in
        "math")
            DATASET_FILE="$BENCHMARK_ROOT/math_test500_dataset.json"
            ;;
        "gsm8k")
            DATASET_FILE="$BENCHMARK_ROOT/gsm8k_test1319_dataset.json"
            ;;
        "gsm8ktoy")
            DATASET_FILE="$BENCHMARK_ROOT/gsm8k_toy20_dataset.json"
            ;;
        "mathtoy")
            DATASET_FILE="$BENCHMARK_ROOT/math_toy20_dataset.json"
            ;;
        "math100")
            DATASET_FILE="$BENCHMARK_ROOT/math_100_dataset.json"
            ;;
        "gsm8k100")
            DATASET_FILE="$BENCHMARK_ROOT/gsm8k_100_dataset.json"
            ;;
        "gsm8k500")
            DATASET_FILE="$BENCHMARK_ROOT/gsm8k_500_dataset.json"
            ;;
        *)
            echo "Error: Unsupported dataset name: $DATASET_NAME"
            echo "Supported datasets: math, gsm8k, gsm8ktoy, mathtoy, math100, gsm8k100, gsm8k500"
            exit 1
            ;;
    esac
    
    # Check if dataset file exists
    if [ ! -f "$DATASET_FILE" ]; then
        echo "Error: Dataset file not found: $DATASET_FILE"
        exit 1
    fi
    
    echo "Using dataset file: $DATASET_FILE"
    
    # Partition the dataset
    python3 scripts/partition_dataset.py "$DATASET_FILE" $GPU_COUNT --output-dir $BENCHMARK_ROOT/temp_partitions
    
    echo "Dataset partitioned into $GPU_COUNT parts"
}

# Run experiment on a single GPU
run_single_gpu_experiment() {
    local gpu_id=$1
    local partition_file=$2
    local output_dir=$3
    
    echo "Starting experiment on GPU $gpu_id with partition: $partition_file"
    
    # Set GPU environment
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    # Create a GPU-specific dataset file
    local gpu_dataset="$BENCHMARK_ROOT/${DATASET_NAME}_gpu_${gpu_id}_dataset.json"
    cp "$partition_file" "$gpu_dataset"
    
    # Create a temporary Python script that modifies the dataset loading
    local temp_main="temp_main_gpu_${gpu_id}.py"
    cat > "$temp_main" << EOF
import sys
import os
sys.path.insert(0, '.')

# Modify the dataset loading to use our GPU-specific file
import inferenceKit.data_config as data_config
from functools import partial

# Store original supported_dataset
original_supported_dataset = data_config.supported_dataset.copy()

# Get the original dataset class from the partial function
original_dataset_func = data_config.supported_dataset['$DATASET_NAME']
original_dataset_class = original_dataset_func.func

# Create custom dataset class that loads our partition
class CustomDataset(original_dataset_class):
    def __init__(self, dataset_name='$DATASET_NAME', dataset_path='$gpu_dataset', **kwargs):
        # Override dataset_path to use our partition
        super().__init__(dataset_name, dataset_path, **kwargs)

# Replace the dataset in supported_dataset with our custom class
data_config.supported_dataset['$DATASET_NAME'] = partial(CustomDataset, dataset_name='$DATASET_NAME', dataset_path='$gpu_dataset')

# Now run the original main
if __name__ == '__main__':
    from main import main
    main()
EOF
    
    # Run the experiment with the modified main
    echo "Starting GPU $gpu_id experiment in directory: $output_dir"
    python3 "$temp_main" \
        --config $CONFIG_FILE \
        --work-dir $output_dir \
        --exp-name gpu_${gpu_id}_experiment \
        --data $DATASET_NAME \
        --model $MODEL_NAME \
        --reward_model $REWARD_MODEL \
        --dtype bfloat16 \
        --flash-attn \
        --debug 2>&1 | tee "$output_dir/gpu_${gpu_id}_log.txt" &
    
    # Store the PID for this process
    echo $! > "$output_dir/gpu_${gpu_id}.pid"
    
    # Store temporary files for later cleanup
    echo "$temp_main" >> "$output_dir/temp_files.txt"
    echo "$gpu_dataset" >> "$output_dir/temp_files.txt"
}

# Run experiments in parallel across all GPUs
run_parallel_experiments() {
    echo "Starting parallel experiments across $GPU_COUNT GPUs..."
    
    # Prepare all experiments first
    local all_experiments=()
    for i in $(seq 0 $((GPU_COUNT-1))); do
        local output_dir="$WORK_DIR/gpu_${i}"
        
        # Find the actual partition file based on the actual dataset filename
        local actual_partition=""
        case $DATASET_NAME in
            "math")
                actual_partition=$(ls $BENCHMARK_ROOT/temp_partitions/math_test500_dataset_partition_$(printf "%03d" $((i+1))).json 2>/dev/null | head -n1)
                ;;
            "gsm8k")
                actual_partition=$(ls $BENCHMARK_ROOT/temp_partitions/gsm8k_test1319_dataset_partition_$(printf "%03d" $((i+1))).json 2>/dev/null | head -n1)
                ;;
            "gsm8ktoy")
                actual_partition=$(ls $BENCHMARK_ROOT/temp_partitions/gsm8k_toy20_dataset_partition_$(printf "%03d" $((i+1))).json 2>/dev/null | head -n1)
                ;;
            "mathtoy")
                actual_partition=$(ls $BENCHMARK_ROOT/temp_partitions/math_toy20_dataset_partition_$(printf "%03d" $((i+1))).json 2>/dev/null | head -n1)
                ;;
            "math100")
                actual_partition=$(ls $BENCHMARK_ROOT/temp_partitions/math_100_dataset_partition_$(printf "%03d" $((i+1))).json 2>/dev/null | head -n1)
                ;;
            "gsm8k100")
                actual_partition=$(ls $BENCHMARK_ROOT/temp_partitions/gsm8k_100_dataset_partition_$(printf "%03d" $((i+1))).json 2>/dev/null | head -n1)
                ;;
            "gsm8k500")
                actual_partition=$(ls $BENCHMARK_ROOT/temp_partitions/gsm8k_500_dataset_partition_$(printf "%03d" $((i+1))).json 2>/dev/null | head -n1)
                ;;
        esac
        
        if [ -z "$actual_partition" ]; then
            echo "Error: Could not find partition file for GPU $i"
            echo "Looking for partition $(printf "%03d" $((i+1))) for dataset $DATASET_NAME"
            echo "Available partition files:"
            ls $BENCHMARK_ROOT/temp_partitions/*.json 2>/dev/null || echo "No partition files found"
            exit 1
        fi
        
        # Store experiment details for parallel execution
        all_experiments+=("$i:$actual_partition:$output_dir")
    done
    
    # Start all experiments simultaneously
    echo "Starting all $GPU_COUNT experiments simultaneously..."
    for experiment in "${all_experiments[@]}"; do
        IFS=':' read -r gpu_id partition_file output_dir <<< "$experiment"
        run_single_gpu_experiment $gpu_id "$partition_file" "$output_dir"
    done
    
    echo "All experiments started. Waiting for completion..."
    
    # Wait for all experiments to complete
    wait_for_completion
    
    echo "All experiments completed!"
}

# Wait for all experiments to complete
wait_for_completion() {
    local all_pids=()
    
    # Collect all PIDs
    for i in $(seq 0 $((GPU_COUNT-1))); do
        local pid_file="$WORK_DIR/gpu_${i}/gpu_${i}.pid"
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            all_pids+=($pid)
            echo "GPU $i PID: $pid"
        fi
    done
    
    echo "Monitoring parallel execution..."
    echo "All $GPU_COUNT processes are running simultaneously"
    
    # Wait for all processes
    for pid in "${all_pids[@]}"; do
        if [ ! -z "$pid" ]; then
            wait $pid
        fi
    done
    
    echo "All processes completed"
}

# Combine results from all GPUs
combine_results() {
    echo "Combining results from all GPUs..."
    
    # Collect all experiment directories and debug their contents
    local exp_dirs=()
    for i in $(seq 0 $((GPU_COUNT-1))); do
        local gpu_dir="$WORK_DIR/gpu_${i}"
        if [ -d "$gpu_dir" ]; then
            # Look for the actual experiment directory (nested structure)
            local actual_exp_dir=""
            local nested_dir="$gpu_dir/${DATASET_NAME}-${MODEL_NAME}-dpts"
            
            if [ -d "$nested_dir" ]; then
                # Try multiple patterns to find the experiment subdirectory
                local exp_subdir=""
                
                # Pattern 1: gpu_X_experiment* (find the most recent one)
                local candidates=$(find "$nested_dir" -name "gpu_${i}_experiment*" -type d 2>/dev/null)
                if [ ! -z "$candidates" ]; then
                    exp_subdir=$(echo "$candidates" | xargs ls -td 2>/dev/null | head -n1)
                fi
                
                # Pattern 2: any directory containing "experiment" (find the most recent one)
                if [ -z "$exp_subdir" ]; then
                    candidates=$(find "$nested_dir" -name "*experiment*" -type d 2>/dev/null)
                    if [ ! -z "$candidates" ]; then
                        exp_subdir=$(echo "$candidates" | xargs ls -td 2>/dev/null | head -n1)
                    fi
                fi
                
                # Pattern 3: any subdirectory (fallback, find the most recent one)
                if [ -z "$exp_subdir" ]; then
                    candidates=$(find "$nested_dir" -maxdepth 1 -type d ! -path "$nested_dir" 2>/dev/null)
                    if [ ! -z "$candidates" ]; then
                        exp_subdir=$(echo "$candidates" | xargs ls -td 2>/dev/null | head -n1)
                    fi
                fi
                
                if [ ! -z "$exp_subdir" ]; then
                    actual_exp_dir="$exp_subdir"
                fi
            fi
            
            if [ ! -z "$actual_exp_dir" ] && [ -d "$actual_exp_dir" ]; then
                exp_dirs+=("$actual_exp_dir")
                echo "Found experiment directory for GPU $i"
            else
                echo "Warning: Could not find experiment directory for GPU $i"
            fi
        else
            echo "Warning: GPU $i directory not found: $gpu_dir"
        fi
    done
    
    if [ ${#exp_dirs[@]} -eq 0 ]; then
        echo "Error: No experiment directories found to combine"
        return 1
    fi
    
    # Simple approach: combine evaluation results from all GPUs
    echo "Combining evaluation results from all GPUs..."
    
    # Create a simple Python script to combine evaluation results
    cat > "$WORK_DIR/combine_evaluation_results.py" << 'EOF'
import sys
import os
import json
from pathlib import Path

def combine_evaluation_results(exp_dirs, dataset_name):
    """Combine evaluation results from all experiment directories."""
    all_evaluation_results = {}
    all_inference_metrics = []
    all_clustering_metrics = []
    
    for exp_dir in exp_dirs:
        print(f"Processing: {exp_dir}")
        
        # Load evaluation results
        eval_file = os.path.join(exp_dir, "evaluation_results.json")
        
        if os.path.exists(eval_file):
            try:
                with open(eval_file, 'r') as f:
                    eval_results = json.load(f)
                print(f"  Loaded evaluation results with {len(eval_results)} voting methods")
                
                # Combine evaluation results for each voting method
                for voting_method, metrics in eval_results.items():
                    if voting_method not in all_evaluation_results:
                        all_evaluation_results[voting_method] = {
                            "total_samples": 0,
                            "correct_samples": 0,
                            "no_match_samples": 0
                        }
                    
                    all_evaluation_results[voting_method]["total_samples"] += metrics.get("total_samples", 0)
                    all_evaluation_results[voting_method]["correct_samples"] += metrics.get("correct_samples", 0)
                    all_evaluation_results[voting_method]["no_match_samples"] += metrics.get("no_match_samples", 0)
                    
            except Exception as e:
                print(f"  Error loading evaluation results: {e}")
        else:
            print(f"  No evaluation_results.json found")
        
        # Load inference metrics
        metrics_file = os.path.join(exp_dir, "inference_metrics.json")
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                all_inference_metrics.append(metrics)
            except Exception as e:
                print(f"  Error loading inference metrics: {e}")
        
        # Load clustering metrics
        clustering_file = os.path.join(exp_dir, "clustering_metrics.json")
        if os.path.exists(clustering_file):
            try:
                with open(clustering_file, 'r') as f:
                    clustering_metrics = json.load(f)
                all_clustering_metrics.append(clustering_metrics)
            except Exception as e:
                print(f"  Error loading clustering metrics: {e}")
    
    # Calculate final accuracies for each voting method
    final_evaluation_results = {}
    for voting_method, metrics in all_evaluation_results.items():
        total = metrics["total_samples"]
        no_match = metrics["no_match_samples"]
        correct = metrics["correct_samples"]
        
        # Calculate accuracy excluding no_match samples
        valid_samples = total - no_match
        if valid_samples > 0:
            accuracy = correct / valid_samples
        else:
            accuracy = 0.0
            
        final_evaluation_results[voting_method] = {
            "accuracy": accuracy,
            "total_samples": total,
            "correct_samples": correct,
            "no_match_samples": no_match,
            "valid_samples": valid_samples
        }
    
    print(f"Combined evaluation results from {len(exp_dirs)} GPU experiments")
    for method, results in final_evaluation_results.items():
        print(f"  {method}: {results['correct_samples']}/{results['valid_samples']} = {results['accuracy']:.4f} (no_match: {results['no_match_samples']})")
    
    # Aggregate simplified clustering metrics
    combined_clustering_metrics = {
        "total_node_explorations": 0,
        "total_nodes_pruned": 0,
        "total_terminated_nodes": 0,
        "total_all_nodes": 0,
        "clustering_applied_count": 0,
        "total_samples": 0
    }
    
    if all_clustering_metrics:
        for metrics in all_clustering_metrics:
            combined_clustering_metrics["total_node_explorations"] += metrics.get("total_node_explorations", 0)
            combined_clustering_metrics["total_nodes_pruned"] += metrics.get("total_nodes_pruned", 0)
            combined_clustering_metrics["total_terminated_nodes"] += metrics.get("total_terminated_nodes", 0)
            combined_clustering_metrics["total_all_nodes"] += metrics.get("total_all_nodes", 0)
            combined_clustering_metrics["clustering_applied_count"] += metrics.get("clustering_applied_count", 0)
            combined_clustering_metrics["total_samples"] += metrics.get("total_samples", 0)
        
        # Calculate nodes saved by clustering (only when clustering was actually applied)
        if combined_clustering_metrics["total_samples"] > 0:
            combined_clustering_metrics["nodes_saved_by_clustering"] = (
                combined_clustering_metrics["total_nodes_pruned"] if combined_clustering_metrics["clustering_applied_count"] > 0 else 0
            )
    
    return final_evaluation_results, all_inference_metrics, combined_clustering_metrics

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python combine_evaluation_results.py <exp_dir1> <exp_dir2> ... <dataset_name>")
        sys.exit(1)
    
    exp_dirs = sys.argv[1:-1]
    dataset_name = sys.argv[-1]
    
    final_eval_results, all_metrics, combined_clustering_metrics = combine_evaluation_results(exp_dirs, dataset_name)
    
    # Save combined results
    output_dir = "combined_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save final evaluation results
    with open(os.path.join(output_dir, "final_evaluation_results.json"), "w") as f:
        json.dump(final_eval_results, f, indent=2, ensure_ascii=False)
    
    # Combine inference metrics if available
    if all_metrics:
        combined_metrics = {
            "inference_time_seconds": sum(m.get("inference_time_seconds", 0) for m in all_metrics),
            "peak_memory_gb": max(m.get("peak_memory_gb", 0) for m in all_metrics),
            "total_questions": sum(m.get("total_questions", 0) for m in all_metrics),
            "source_experiments": len(all_metrics),
            "individual_metrics": all_metrics
        }
        
        if combined_metrics["total_questions"] > 0:
            combined_metrics["avg_time_per_question_seconds"] = (
                combined_metrics["inference_time_seconds"] / combined_metrics["total_questions"]
            )
        
        with open(os.path.join(output_dir, "combined_inference_metrics.json"), "w") as f:
            json.dump(combined_metrics, f, indent=2, ensure_ascii=False)
    
    print(f"Combined results saved to: {output_dir}/")
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL MULTI-GPU EVALUATION RESULTS")
    print("="*70)
    for method, results in final_eval_results.items():
        print(f"{method:12}: {results['accuracy']:.4f} ({results['correct_samples']}/{results['valid_samples']}) [no_match: {results['no_match_samples']}]")
    
    # Add timing information if available
    if all_metrics:
        total_inference_time = sum(m.get("inference_time_seconds", 0) for m in all_metrics)
        total_questions = sum(m.get("total_questions", 0) for m in all_metrics)
        avg_time_per_question = total_inference_time / total_questions if total_questions > 0 else 0
        
        print("-" * 60)
        print(f"Total Inference Time: {total_inference_time:.2f} seconds")
        print(f"Average Time per Question: {avg_time_per_question:.3f} seconds")
        print(f"Total Questions Processed: {total_questions}")
    
    # Add clustering metrics if available
    if combined_clustering_metrics and combined_clustering_metrics.get("total_samples", 0) > 0:
        print("-" * 60)
        print("CLUSTERING EFFICIENCY SUMMARY")
        print("-" * 60)
        print(f"📊 Dataset: {combined_clustering_metrics['total_samples']} samples processed")
        total_samples = combined_clustering_metrics.get('total_samples', 1)
        avg_explorations = combined_clustering_metrics['total_node_explorations'] / total_samples
        avg_nodes = combined_clustering_metrics['total_all_nodes'] / total_samples
        avg_pruned = combined_clustering_metrics.get('total_nodes_pruned', 0) / total_samples
        avg_clustering_applied = combined_clustering_metrics.get('clustering_applied_count', 0) / total_samples
        
        print(f"🌳 Nodes: {combined_clustering_metrics['total_node_explorations']:,} node explorations → {combined_clustering_metrics['total_all_nodes']:,} total nodes in tree")
        print(f"💾 Clustering: {combined_clustering_metrics.get('total_nodes_pruned', 0):,} nodes pruned ({combined_clustering_metrics.get('clustering_applied_count', 0)} times applied)")
        print(f"📊 Averages per sample: {avg_explorations:.1f} explorations, {avg_nodes:.1f} nodes, {avg_pruned:.1f} pruned, {avg_clustering_applied:.1f} clustering applications")
    
    print("="*60)
EOF
    
    # Run the evaluation combination script
    cd "$WORK_DIR"
    
    # Convert absolute paths to relative paths since we're now in $WORK_DIR
    relative_exp_dirs=()
    for exp_dir in "${exp_dirs[@]}"; do
        # Remove the $WORK_DIR prefix to make paths relative
        relative_dir="${exp_dir#$WORK_DIR/}"
        relative_exp_dirs+=("$relative_dir")
    done
    
    python3 combine_evaluation_results.py "${relative_exp_dirs[@]}" "$DATASET_NAME"
    cd - > /dev/null
    
    # Add wall clock time to combined metrics
    if [ -f "$WORK_DIR/combined_results/combined_inference_metrics.json" ]; then
        local current_time=$(date +%s)
        local wall_clock_time=$((current_time - start_time))
        
        # Add wall clock time to the metrics file
        python3 -c "
import json
import sys

# Read the existing metrics
with open('$WORK_DIR/combined_results/combined_inference_metrics.json', 'r') as f:
    metrics = json.load(f)

# Add wall clock time
metrics['total_wall_clock_time_seconds'] = $wall_clock_time
metrics['wall_clock_start_timestamp'] = '$start_timestamp'
metrics['wall_clock_end_timestamp'] = '$(date)'

# Write back to file
with open('$WORK_DIR/combined_results/combined_inference_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

# Wall clock time added to metrics file
"
    fi
    
    echo "Results combined and saved to: $WORK_DIR/combined_results"
}

# Clean up temporary files
cleanup() {
    echo "Cleaning up temporary files..."
    
    # Clean up partition files
    python3 scripts/cleanup_partitions.py --dir $BENCHMARK_ROOT/temp_partitions
    
    # Clean up PID files and temporary files for each GPU
    for i in $(seq 0 $((GPU_COUNT-1))); do
        local gpu_dir="$WORK_DIR/gpu_${i}"
        rm -f "$gpu_dir/gpu_${i}.pid"
        
        # Clean up temporary files listed in temp_files.txt
        if [ -f "$gpu_dir/temp_files.txt" ]; then
            while read -r temp_file; do
                rm -f "$temp_file"
            done < "$gpu_dir/temp_files.txt"
            rm -f "$gpu_dir/temp_files.txt"
        fi
    done
    
    # Clean up temporary files created during combination
    rm -f "$WORK_DIR/combine_evaluation_results.py"
    
    echo "Cleanup completed"
}

# Print usage information
print_usage() {
    echo "Usage: $0 [DATASET_NAME] [MODEL_NAME] [REWARD_MODEL] [EXP_NAME] [WORK_DIR] [CONFIG_FILE]"
    echo ""
    echo "Parameters:"
    echo "  DATASET_NAME  - Dataset to use (default: math)"
    echo "                Supported: math, gsm8k, gsm8ktoy, mathtoy, math100, gsm8k100, gsm8k500"
    echo "  MODEL_NAME    - Model to use (default: qwen-1.5b)"
    echo "  REWARD_MODEL  - Reward model to use (default: mistral_prm-7b)"
    echo "  EXP_NAME      - Experiment name (default: multi_gpu_test)"
    echo "  WORK_DIR      - Working directory (default: ./outputs)"
    echo "  CONFIG_FILE   - Config file path (default: configs/inference/DPTS.json)"
    echo ""
    echo "Dataset Mappings (from data_config.py):"
    echo "  math     -> math_test500_dataset.json"
    echo "  gsm8k    -> gsm8k_test1319_dataset.json"
    echo "  gsm8ktoy -> gsm8k_toy20_dataset.json"
    echo "  mathtoy  -> math_toy20_dataset.json"
    echo "  math100  -> math_100_dataset.json"
    echo "  gsm8k100 -> gsm8k_100_dataset.json"
    echo "  gsm8k500 -> gsm8k_500_dataset.json"
    echo ""
    echo "Examples:"
    echo "  $0 math qwen-1.5b mistral_prm-7b my_experiment ./results"
    echo "  $0 gsm8k qwen-1.5b mistral_prm-7b gsm8k_test ./outputs configs/inference/DPTS.json"
    echo "  $0 gsm8ktoy qwen-1.5b mistral_prm-7b toy_test ./outputs"
    echo "  $0 math100 qwen-1.5b mistral_prm-7b math100_test ./outputs"
    echo "  $0 gsm8k100 qwen-1.5b mistral_prm-7b gsm8k100_test ./outputs"
    echo "  $0 gsm8k500 qwen-1.5b mistral_prm-7b gsm8k500_test ./outputs"
}

# Main execution
main() {
    # Record start time
    local start_time=$(date +%s)
    local start_timestamp=$(date)
    
    echo "=========================================="
    echo "Multi-GPU Experiment Runner"
    echo "=========================================="
    echo "Dataset: $DATASET_NAME"
    echo "Model: $MODEL_NAME"
    echo "Reward Model: $REWARD_MODEL"
    echo "Experiment Name: $EXP_NAME"
    echo "Work Directory: $WORK_DIR"
    echo "Config File: $CONFIG_FILE"
    echo "Start Time: $start_timestamp"
    echo "=========================================="
    
    # Detect GPUs
    detect_gpus
    
    # Setup
    setup_experiment_dirs
    partition_dataset
    
    # Run experiments
    run_parallel_experiments
    
    # Combine results
    combine_results
    
    # Cleanup
    cleanup
    
    # Calculate and display total wall clock time
    local end_time=$(date +%s)
    local end_timestamp=$(date)
    local total_wall_time=$((end_time - start_time))
    local hours=$((total_wall_time / 3600))
    local minutes=$(((total_wall_time % 3600) / 60))
    local seconds=$((total_wall_time % 60))
    
    echo "=========================================="
    echo "Multi-GPU experiment completed successfully!"
    echo "=========================================="
    echo "Start Time: $start_timestamp"
    echo "End Time:   $end_timestamp"
    echo "Total Wall Clock Time: ${hours}h ${minutes}m ${seconds}s"
    echo "Results saved to: $WORK_DIR/combined_results"
    echo "=========================================="
}

# Handle command line arguments
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    print_usage
    exit 0
fi

# Run main function
main "$@"
