#!/usr/bin/env python3
"""
Hyperparameter Analysis Script
===========================

This script performs hyperparameter analysis on any parameter in the config file.
It can test any parameter with different value ranges and collect timing and accuracy data.

Usage Options:

1. Use configuration from script (RECOMMENDED):
    python scripts/hyperparameter_analysis.py
    # or
    python scripts/hyperparameter_analysis.py --use-config

2. Use command line arguments:
    python scripts/hyperparameter_analysis.py --parameter clustering_threshold --min-value 0.0 --max-value 1.0 --step-size 0.1

Configuration:
    Edit the HYPERPARAMETER_CONFIG dictionary at the top of this script to set your parameters.

Examples:
    # Test clustering threshold (using script config)
    python scripts/hyperparameter_analysis.py

    # Test tree_width (using command line)
    python scripts/hyperparameter_analysis.py --parameter tree_width --min-value 2 --max-value 8 --step-size 1

    # Test lambda_es (using command line)
    python scripts/hyperparameter_analysis.py --parameter lambda_es --min-value 0.1 --max-value 1.0 --step-size 0.1
"""

import os
import sys
import json
import csv
import time
import subprocess
import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Any, Union
import tempfile

# =============================================================================
# CONFIGURATION - Set your parameters here instead of using command line args
# =============================================================================

# Hyperparameter Analysis Configuration
# ====================================
# Edit these values to configure your hyperparameter analysis
# Then simply run: python scripts/hyperparameter_analysis.py

HYPERPARAMETER_CONFIG = {
    # Parameter to test
    'parameter_name': 'clustering_threshold',  # Options: clustering_threshold, tree_width, tree_depth, lambda_es, lambda_ds, etc.
    
    # Value range for the parameter
    'min_value': 0.0,
    'max_value': 1.0,
    'step_size': 1.0,
    
    # Experiment settings
    'dataset': 'mathtoy',  # Options: mathtoy, gsm8k100, math100, gsm8k500, etc.
    'model': 'qwen-1.5b',  # Options: qwen-1.5b, qwen-7b, etc.
    'reward_model': 'mistral_prm-7b',
    
    # File paths
    'base_config': 'configs/inference/SSDP.json',
    'work_dir': './hyperparameter_analysis',
    'output_csv': None,  # None = auto-generate filename
}

# Example configurations for different parameters:
# ===============================================

# For testing tree_width:
# HYPERPARAMETER_CONFIG = {
#     'parameter_name': 'tree_width',
#     'min_value': 2,
#     'max_value': 8,
#     'step_size': 1,
#     'dataset': 'mathtoy',
#     'model': 'qwen-1.5b',
#     'reward_model': 'mistral_prm-7b',
#     'base_config': 'configs/inference/SSDP.json',
#     'work_dir': './hyperparameter_analysis',
#     'output_csv': None,
# }

# For testing lambda_es:
# HYPERPARAMETER_CONFIG = {
#     'parameter_name': 'lambda_es',
#     'min_value': 0.1,
#     'max_value': 1.0,
#     'step_size': 0.1,
#     'dataset': 'mathtoy',
#     'model': 'qwen-1.5b',
#     'reward_model': 'mistral_prm-7b',
#     'base_config': 'configs/inference/SSDP.json',
#     'work_dir': './hyperparameter_analysis',
#     'output_csv': None,
# }

# =============================================================================

class HyperparameterAnalyzer:
    def __init__(self, 
                 base_config: str = "configs/inference/SSDP.json",
                 parameter_name: str = "clustering_threshold",
                 value_range: tuple = (0.0, 1.0),
                 step_size: Union[float, int] = 0.1,
                 dataset: str = "mathtoy",
                 model: str = "qwen-1.5b",
                 reward_model: str = "mistral_prm-7b",
                 work_dir: str = "./hyperparameter_analysis"):
        """
        Initialize the hyperparameter analyzer.
        
        Args:
            base_config: Path to the base configuration file
            parameter_name: Name of the parameter to test (e.g., 'clustering_threshold', 'tree_width')
            value_range: Tuple of (min, max) values to test
            step_size: Step size between values
            dataset: Dataset to use for experiments
            model: Model to use for experiments
            reward_model: Reward model to use
            work_dir: Base directory for all experiments
        """
        self.base_config = base_config
        self.parameter_name = parameter_name
        self.value_range = value_range
        self.step_size = step_size
        self.dataset = dataset
        self.model = model
        self.reward_model = reward_model
        # Make work directory absolute to avoid path issues
        self.work_dir = os.path.abspath(work_dir)
        
        # Generate parameter values
        self.parameter_values = self._generate_parameter_values()
        
        # Results storage
        self.results = []
        
        # Ensure work directory exists
        os.makedirs(self.work_dir, exist_ok=True)
        
    def _generate_parameter_values(self) -> List[Union[float, int]]:
        """Generate list of parameter values to test."""
        min_val, max_val = self.value_range
        values = []
        current = min_val
        
        # Determine if we're working with integers or floats
        if isinstance(self.step_size, int) or (isinstance(self.step_size, float) and self.step_size.is_integer()):
            # Integer values
            while current <= max_val:
                values.append(int(current))
                current += self.step_size
        else:
            # Float values
            while current <= max_val:
                values.append(round(current, 3))
                current += self.step_size
        return values
    
    def _create_modified_config(self, parameter_value: Union[float, int], output_path: str) -> str:
        """Create a modified config file with the specified parameter value."""
        # Load base config
        with open(self.base_config, 'r') as f:
            config = json.load(f)
        
        # Modify the specified parameter
        config['config'][self.parameter_name] = parameter_value
        
        # Save modified config
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        return output_path
    
    def _run_experiment(self, parameter_value: Union[float, int], exp_name: str) -> Dict[str, Any]:
        """Run a single experiment with the given parameter value."""
        print(f"\n{'='*60}")
        print(f"Running experiment with {self.parameter_name} = {parameter_value}")
        print(f"{'='*60}")
        
        # Create modified config
        config_path = os.path.join(self.work_dir, f"config_{self.parameter_name}_{parameter_value}.json")
        self._create_modified_config(parameter_value, config_path)
        
        # Create experiment directory
        exp_dir = os.path.join(self.work_dir, f"exp_{self.parameter_name}_{parameter_value}")
        os.makedirs(exp_dir, exist_ok=True)
        
        # Make paths absolute for the multi-GPU script
        config_path = os.path.abspath(config_path)
        exp_dir = os.path.abspath(exp_dir)
        
        # Always use multi-GPU script (it handles single GPU case too)
        cmd = [
            "bash", "scripts/multi_gpu_run.sh",
            self.dataset,
            self.model,
            self.reward_model,
            exp_name,
            exp_dir,
            config_path
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Record start time
        start_time = time.time()
        
        try:
            # Run experiment from project root directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(script_dir, '..'))
            print(f"Script directory: {script_dir}")
            print(f"Project root: {project_root}")
            print(f"Working directory: {os.getcwd()}")
            print(f"Dataset file exists: {os.path.exists(os.path.join(project_root, 'benchmark', 'math_toy20_dataset.json'))}")
            
            # Change to project root directory before running
            original_cwd = os.getcwd()
            os.chdir(project_root)
            print(f"Changed working directory to: {os.getcwd()}")
            
            result = subprocess.run(cmd, 
                                  capture_output=True, 
                                  text=True)
            
            # Restore original working directory
            os.chdir(original_cwd)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            if result.returncode != 0:
                print(f"Experiment failed with return code {result.returncode}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return {
                    'parameter_value': parameter_value,
                    'success': False,
                    'error': result.stderr,
                    'total_time': total_time
                }
            
            # Extract results
            metrics = self._extract_metrics(exp_dir, parameter_value, total_time)
            metrics['parameter_value'] = parameter_value
            metrics['success'] = True
            
            return metrics
            
        except Exception as e:
            end_time = time.time()
            total_time = end_time - start_time
            print(f"Experiment failed with exception: {e}")
            return {
                'parameter_value': parameter_value,
                'success': False,
                'error': str(e),
                'total_time': total_time
            }
    
    def _extract_metrics(self, exp_dir: str, parameter_value: Union[float, int], total_time: float) -> Dict[str, Any]:
        """Extract metrics from experiment results."""
        metrics = {
            'parameter_value': parameter_value,
            'accuracy': 0.0,
            'avg_inference_time_per_sample': 0.0
        }
        
        # Look for results in different possible locations
        possible_result_dirs = [
            os.path.join(exp_dir, "combined_results"),  # Multi-GPU combined results
            exp_dir,
            os.path.join(exp_dir, f"{self.dataset}-{self.model}-dpts"),
        ]
        
        # Find the actual results directory
        results_dir = None
        print(f"Debug: Looking for results in exp_dir: {exp_dir}")
        for possible_dir in possible_result_dirs:
            print(f"Debug: Checking possible_dir: {possible_dir}")
            if os.path.exists(possible_dir):
                print(f"Debug: Found directory: {possible_dir}")
                # Look for subdirectories that might contain results (exclude .ipynb_checkpoints)
                subdirs = [d for d in os.listdir(possible_dir) if os.path.isdir(os.path.join(possible_dir, d)) and d != '.ipynb_checkpoints']
                print(f"Debug: Subdirectories found: {subdirs}")
                if subdirs:
                    # Use the most recent subdirectory
                    subdirs.sort(key=lambda x: os.path.getmtime(os.path.join(possible_dir, x)), reverse=True)
                    results_dir = os.path.join(possible_dir, subdirs[0])
                    print(f"Debug: Using subdirectory: {results_dir}")
                    break
                else:
                    results_dir = possible_dir
                    print(f"Debug: Using directory directly: {results_dir}")
                    break
            else:
                print(f"Debug: Directory does not exist: {possible_dir}")
        
        if not results_dir:
            print(f"Warning: Could not find results directory for {self.parameter_name} {parameter_value}")
            print(f"Debug: exp_dir contents: {os.listdir(exp_dir) if os.path.exists(exp_dir) else 'exp_dir does not exist'}")
            return metrics
        
        # First try to load combined results (multi-GPU output)
        # Check for different possible combined results files
        possible_combined_files = [
            os.path.join(results_dir, "combined_results.json"),
            os.path.join(results_dir, "final_evaluation_results.json"),
            os.path.join(results_dir, "combined_inference_metrics.json"),
            os.path.join(results_dir, "inference_metrics.json")
        ]
        
        combined_file = None
        for file_path in possible_combined_files:
            print(f"Debug: Looking for {os.path.basename(file_path)} at: {file_path}")
            if os.path.exists(file_path):
                print(f"Debug: Found {os.path.basename(file_path)}")
                combined_file = file_path
                break
        
        if combined_file:
            try:
                with open(combined_file, 'r') as f:
                    combined_results = json.load(f)
                
                print(f"Debug: Combined results keys: {list(combined_results.keys())}")
                
                # Handle final_evaluation_results.json format (multiple voting methods)
                if any(key in combined_results for key in ['min_max', 'last_max', 'majority_vote', 'min_vote', 'last_vote']):
                    print(f"Debug: Found final_evaluation_results.json format")
                    # Get accuracy from the first available voting method
                    for method, results in combined_results.items():
                        if isinstance(results, dict) and 'accuracy' in results:
                            metrics['accuracy'] = results['accuracy']
                            print(f"Debug: Found accuracy from {method}: {metrics['accuracy']}")
                            break
                
                # Handle combined_results.json format (evaluation and inference sections)
                elif 'evaluation' in combined_results:
                    eval_data = combined_results['evaluation']
                    print(f"Debug: Evaluation data keys: {list(eval_data.keys()) if isinstance(eval_data, dict) else 'Not a dict'}")
                    if 'accuracy' in eval_data:
                        metrics['accuracy'] = eval_data['accuracy']
                        print(f"Debug: Found accuracy: {metrics['accuracy']}")
                
                # Extract inference time from inference section (if available)
                if 'inference' in combined_results:
                    inference_data = combined_results['inference']
                    print(f"Debug: Inference data keys: {list(inference_data.keys()) if isinstance(inference_data, dict) else 'Not a dict'}")
                    if 'avg_time_per_question_seconds' in inference_data:
                        metrics['avg_inference_time_per_sample'] = inference_data['avg_time_per_question_seconds']
                        print(f"Debug: Found avg_time_per_question_seconds: {metrics['avg_inference_time_per_sample']}")
                
                # If we found accuracy but not timing, look for timing in other files
                if metrics['accuracy'] > 0 and metrics['avg_inference_time_per_sample'] == 0:
                    print(f"Debug: Found accuracy but no timing data, looking for timing files...")
                    timing_files = [
                        os.path.join(results_dir, "combined_inference_metrics.json"),
                        os.path.join(results_dir, "inference_metrics.json"),
                        os.path.join(results_dir, "combined_results.json")
                    ]
                    
                    for timing_file in timing_files:
                        if os.path.exists(timing_file):
                            print(f"Debug: Found timing file: {os.path.basename(timing_file)}")
                            try:
                                with open(timing_file, 'r') as f:
                                    timing_data = json.load(f)
                                
                                # Look for timing data in different possible structures
                                if 'avg_time_per_question_seconds' in timing_data:
                                    metrics['avg_inference_time_per_sample'] = timing_data['avg_time_per_question_seconds']
                                    print(f"Debug: Found timing in {os.path.basename(timing_file)}: {metrics['avg_inference_time_per_sample']}")
                                    break
                                elif 'inference' in timing_data and 'avg_time_per_question_seconds' in timing_data['inference']:
                                    metrics['avg_inference_time_per_sample'] = timing_data['inference']['avg_time_per_question_seconds']
                                    print(f"Debug: Found timing in {os.path.basename(timing_file)}: {metrics['avg_inference_time_per_sample']}")
                                    break
                                elif 'individual_metrics' in timing_data and timing_data['individual_metrics']:
                                    # Calculate average from individual metrics if available
                                    individual_times = [item.get('avg_time_per_question_seconds', 0) for item in timing_data['individual_metrics'] if 'avg_time_per_question_seconds' in item]
                                    if individual_times:
                                        metrics['avg_inference_time_per_sample'] = sum(individual_times) / len(individual_times)
                                        print(f"Debug: Calculated average timing from {len(individual_times)} individual metrics: {metrics['avg_inference_time_per_sample']}")
                                        break
                            except Exception as e:
                                print(f"Debug: Could not load timing from {os.path.basename(timing_file)}: {e}")
                
                print(f"✓ Loaded metrics from {os.path.basename(combined_file)}: accuracy={metrics['accuracy']:.4f}, time={metrics['avg_inference_time_per_sample']:.3f}s")
                return metrics
            except Exception as e:
                print(f"Warning: Could not load combined results from {os.path.basename(combined_file)}: {e}")
        else:
            print(f"Debug: No combined results files found")
            print(f"Debug: Files in results_dir: {os.listdir(results_dir) if os.path.exists(results_dir) else 'results_dir does not exist'}")
        
        # Fallback: Load individual evaluation results for accuracy
        eval_file = os.path.join(results_dir, "evaluation_results.json")
        if os.path.exists(eval_file):
            try:
                with open(eval_file, 'r') as f:
                    eval_results = json.load(f)
                
                # Get accuracy from the first available voting method
                for method, results in eval_results.items():
                    if 'accuracy' in results:
                        metrics['accuracy'] = results['accuracy']
                        break
            except Exception as e:
                print(f"Warning: Could not load evaluation results: {e}")
        
        # Fallback: Load individual inference metrics for average time per sample
        inference_file = os.path.join(results_dir, "inference_metrics.json")
        if os.path.exists(inference_file):
            try:
                with open(inference_file, 'r') as f:
                    inference_metrics = json.load(f)
                
                # Extract average inference time per sample
                metrics['avg_inference_time_per_sample'] = inference_metrics.get('avg_time_per_question_seconds', 0.0)
            except Exception as e:
                print(f"Warning: Could not load inference metrics: {e}")
        
        return metrics
    
    def run_analysis(self) -> List[Dict[str, Any]]:
        """Run the complete hyperparameter analysis."""
        print(f"Starting hyperparameter analysis for {self.parameter_name}")
        print(f"Parameter range: {self.value_range[0]} to {self.value_range[1]} (step: {self.step_size})")
        print(f"Number of experiments: {len(self.parameter_values)}")
        print(f"Dataset: {self.dataset}, Model: {self.model}")
        print(f"Work directory: {self.work_dir}")
        
        self.results = []
        
        for i, parameter_value in enumerate(self.parameter_values):
            print(f"\nProgress: {i+1}/{len(self.parameter_values)}")
            
            exp_name = f"{self.parameter_name}_{parameter_value}"
            result = self._run_experiment(parameter_value, exp_name)
            self.results.append(result)
            
            # Print intermediate results
            if result['success']:
                print(f"✓ {self.parameter_name} {parameter_value}: Accuracy={result.get('accuracy', 0):.4f}, "
                      f"Avg Time/Sample={result.get('avg_inference_time_per_sample', 0):.3f}s")
            else:
                print(f"✗ {self.parameter_name} {parameter_value}: Failed - {result.get('error', 'Unknown error')}")
        
        return self.results
    
    def export_to_csv(self, output_file: str = None) -> str:
        """Export results to CSV file."""
        if not self.results:
            raise ValueError("No results to export. Run analysis first.")
        
        if output_file is None:
            output_file = os.path.join(self.work_dir, f"hyperparameter_analysis_{self.parameter_name}_results.csv")
        
        # Define CSV columns
        fieldnames = [
            'parameter_value',
            'success',
            'accuracy',
            'avg_inference_time_per_sample',
            'error'
        ]
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                # Ensure all fields are present
                row = {field: result.get(field, '') for field in fieldnames}
                writer.writerow(row)
        
        print(f"\nResults exported to: {output_file}")
        return output_file
    
    def print_summary(self):
        """Print a summary of the analysis results."""
        if not self.results:
            print("No results to summarize.")
            return
        
        successful_results = [r for r in self.results if r.get('success', False)]
        failed_results = [r for r in self.results if not r.get('success', False)]
        
        print(f"\n{'='*60}")
        print(f"HYPERPARAMETER ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"Parameter: {self.parameter_name}")
        print(f"Total experiments: {len(self.results)}")
        print(f"Successful: {len(successful_results)}")
        print(f"Failed: {len(failed_results)}")
        
        if successful_results:
            print(f"\nSuccessful Results:")
            print(f"{'Value':<15} {'Accuracy':<10} {'Avg Time/Sample(s)':<18}")
            print(f"{'-'*45}")
            
            for result in successful_results:
                print(f"{result['parameter_value']:<15} "
                      f"{result.get('accuracy', 0):<10.4f} "
                      f"{result.get('avg_inference_time_per_sample', 0):<18.3f}")
        
        if failed_results:
            print(f"\nFailed Results:")
            for result in failed_results:
                print(f"{self.parameter_name} {result['parameter_value']}: {result.get('error', 'Unknown error')}")


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter analysis for any config parameter')
    parser.add_argument('--base-config', default=None,
                       help='Base configuration file (default: from config)')
    parser.add_argument('--parameter', default=None,
                       help='Parameter name to test (e.g., clustering_threshold, tree_width, lambda_es)')
    parser.add_argument('--min-value', type=float, default=None,
                       help='Minimum parameter value')
    parser.add_argument('--max-value', type=float, default=None,
                       help='Maximum parameter value')
    parser.add_argument('--step-size', type=float, default=None,
                       help='Step size for parameter values')
    parser.add_argument('--dataset', default=None,
                       help='Dataset to use (default: from config)')
    parser.add_argument('--model', default=None,
                       help='Model to use (default: from config)')
    parser.add_argument('--reward-model', default=None,
                       help='Reward model to use (default: from config)')
    parser.add_argument('--work-dir', default=None,
                       help='Work directory for experiments (default: from config)')
    parser.add_argument('--output-csv', default=None,
                       help='Output CSV file path (default: auto-generated)')
    parser.add_argument('--use-config', action='store_true',
                       help='Use configuration from script instead of command line args')
    
    args = parser.parse_args()
    
    # Use configuration from script if --use-config is specified or no args provided
    use_script_config = args.use_config or (args.parameter is None and args.min_value is None)
    
    if use_script_config:
        print("="*80)
        print("Using configuration from script:")
        print("="*80)
        for key, value in HYPERPARAMETER_CONFIG.items():
            print(f"  {key}: {value}")
        print("="*80)
        
        # Use script configuration
        parameter_name = HYPERPARAMETER_CONFIG['parameter_name']
        min_value = HYPERPARAMETER_CONFIG['min_value']
        max_value = HYPERPARAMETER_CONFIG['max_value']
        step_size = HYPERPARAMETER_CONFIG['step_size']
        dataset = HYPERPARAMETER_CONFIG['dataset']
        model = HYPERPARAMETER_CONFIG['model']
        reward_model = HYPERPARAMETER_CONFIG['reward_model']
        base_config = HYPERPARAMETER_CONFIG['base_config']
        work_dir = HYPERPARAMETER_CONFIG['work_dir']
        output_csv = HYPERPARAMETER_CONFIG['output_csv']
    else:
        # Use command line arguments
        parameter_name = args.parameter
        min_value = args.min_value
        max_value = args.max_value
        step_size = args.step_size
        dataset = args.dataset or 'mathtoy'
        model = args.model or 'qwen-1.5b'
        reward_model = args.reward_model or 'mistral_prm-7b'
        base_config = args.base_config or 'configs/inference/SSDP.json'
        work_dir = args.work_dir or './hyperparameter_analysis'
        output_csv = args.output_csv
    
    # Validate arguments - handle relative paths from script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(base_config):
        # If it's a relative path, make it relative to the script directory
        base_config = os.path.join(script_dir, '..', base_config)
        base_config = os.path.normpath(base_config)
    
    if not os.path.exists(base_config):
        print(f"Error: Base config file not found: {base_config}")
        print(f"Script directory: {script_dir}")
        print(f"Looking for: {base_config}")
        sys.exit(1)
    
    if min_value >= max_value:
        print("Error: min_value must be less than max_value")
        sys.exit(1)
    
    if step_size <= 0:
        print("Error: step_size must be positive")
        sys.exit(1)
    
    # Create analyzer
    analyzer = HyperparameterAnalyzer(
        base_config=base_config,
        parameter_name=parameter_name,
        value_range=(min_value, max_value),
        step_size=step_size,
        dataset=dataset,
        model=model,
        reward_model=reward_model,
        work_dir=work_dir
    )
    
    try:
        # Run analysis
        results = analyzer.run_analysis()
        
        # Export to CSV
        csv_file = analyzer.export_to_csv(output_csv)
        
        # Print summary
        analyzer.print_summary()
        
        print(f"\nAnalysis complete! Results saved to: {csv_file}")
        print(f"You can now create graphs using the CSV data to visualize the relationship")
        print(f"between {parameter_name}, time, and accuracy.")
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Analysis failed with error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
