#!/usr/bin/env python3
"""
Combine and Analyze Multiple Experiment Results

This script combines inference metrics and results from multiple experiment directories,
calculates combined metrics, merges results, and evaluates the combined dataset.

Usage:
    python combine_results.py <experiment_dirs> [options]

Examples:
    python combine_results.py outputs/exp1 outputs/exp2 outputs/exp3
    python combine_results.py outputs/* --output-dir combined_results
    python combine_results.py outputs/exp1 outputs/exp2 --evaluate-only
"""

import argparse
import json
import os
import glob
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import datetime
from collections import defaultdict


def load_inference_metrics(metrics_file: str) -> Dict[str, Any]:
    """Load inference metrics from a JSON file."""
    try:
        with open(metrics_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Inference metrics file not found: {metrics_file}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in metrics file {metrics_file}: {e}")


def load_results_file(results_file: str) -> List[Dict[str, Any]]:
    """Load results from a JSON file."""
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Results file not found: {results_file}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in results file {results_file}: {e}")


def combine_inference_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine multiple inference metrics into a single metrics object."""
    if not metrics_list:
        raise ValueError("No metrics to combine")
    
    # Initialize combined metrics
    combined = {
        "inference_time_seconds": 0.0,
        "peak_memory_gb": 0.0,
        "total_questions": 0,
        "avg_time_per_question_seconds": 0.0,
        "start_time": float('inf'),
        "end_time": 0.0,
        "timestamp": datetime.datetime.now().isoformat(),
        "source_experiments": [],
        "config": None
    }
    
    total_time = 0.0
    total_questions = 0
    peak_memory = 0.0
    
    for i, metrics in enumerate(metrics_list):
        # Sum up times and questions
        total_time += metrics.get("inference_time_seconds", 0.0)
        total_questions += metrics.get("total_questions", 0)
        
        # Track peak memory (maximum across all experiments)
        peak_memory = max(peak_memory, metrics.get("peak_memory_gb", 0.0))
        
        # Track time range
        combined["start_time"] = min(combined["start_time"], metrics.get("start_time", float('inf')))
        combined["end_time"] = max(combined["end_time"], metrics.get("end_time", 0.0))
        
        # Store source experiment info
        combined["source_experiments"].append({
            "experiment_id": i + 1,
            "inference_time_seconds": metrics.get("inference_time_seconds", 0.0),
            "total_questions": metrics.get("total_questions", 0),
            "peak_memory_gb": metrics.get("peak_memory_gb", 0.0),
            "timestamp": metrics.get("timestamp", "unknown")
        })
        
        # Use config from first experiment (assuming they're similar)
        if combined["config"] is None:
            combined["config"] = metrics.get("config", {})
    
    # Calculate combined metrics
    combined["inference_time_seconds"] = total_time
    combined["total_questions"] = total_questions
    combined["peak_memory_gb"] = peak_memory
    combined["avg_time_per_question_seconds"] = total_time / total_questions if total_questions > 0 else 0.0
    
    return combined


def combine_results_files(results_files: List[str]) -> List[Dict[str, Any]]:
    """Combine multiple results files into a single results list."""
    if not results_files:
        raise ValueError("No results files to combine")
    
    combined_results = []
    seen_indices = set()
    
    for results_file in results_files:
        try:
            results = load_results_file(results_file)
            for result in results:
                # Avoid duplicates based on index
                if result.get("index") not in seen_indices:
                    combined_results.append(result)
                    seen_indices.add(result.get("index"))
        except Exception as e:
            print(f"Warning: Could not load results file {results_file}: {e}")
            continue
    
    # Sort by index to maintain order
    combined_results.sort(key=lambda x: x.get("index", 0))
    
    return combined_results


def evaluate_results(results: List[Dict[str, Any]], dataset_name: str = None) -> Dict[str, Any]:
    """Evaluate combined results using the dataset evaluation logic."""
    # Import the dataset evaluation logic
    try:
        import sys
        import os
        # Add parent directory to path to import inferenceKit
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
            
        from inferenceKit.data_config import supported_dataset
        from inferenceKit.dataset.basedataset import BaseDataset
        
        if dataset_name and dataset_name in supported_dataset:
            dataset = supported_dataset[dataset_name]()
        else:
            # Create a dummy dataset for evaluation
            dataset = BaseDataset("combined", "")
        
        # Use the dataset's evaluation method
        return dataset.evaluate_results(results, print_result=False, print_no_match=False)
        
    except ImportError:
        # Fallback evaluation if inferenceKit is not available
        print("Warning: Could not import inferenceKit for evaluation. Using basic evaluation.")
        return basic_evaluate_results(results)


def basic_evaluate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Basic evaluation when inferenceKit is not available."""
    total_samples = len(results)
    correct_samples = 0
    no_match_samples = 0
    
    print(f"Basic evaluation: Processing {total_samples} results")
    
    for i, result in enumerate(results):
        response = str(result.get('response', ''))
        answer = str(result.get('answer', ''))
        
        print(f"Sample {i+1}:")
        print(f"  Response: '{response[:100]}...' (length: {len(response)})")
        print(f"  Answer: '{answer}'")
        
        # Simple string matching for basic evaluation
        if response.strip().lower() == answer.strip().lower():
            correct_samples += 1
            print(f"  Result: CORRECT")
        elif not response.strip():
            no_match_samples += 1
            print(f"  Result: NO MATCH (empty response)")
        else:
            print(f"  Result: INCORRECT")
        print()
    
    accuracy = correct_samples / total_samples if total_samples > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "total_samples": total_samples,
        "correct_samples": correct_samples,
        "no_match_samples": no_match_samples
    }


def find_experiment_files(experiment_dir: str) -> Dict[str, str]:
    """Find inference_metrics.json and results files in an experiment directory."""
    experiment_path = Path(experiment_dir)
    
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    
    files = {
        "metrics": None,
        "results": []
    }
    
    # Find inference_metrics.json
    metrics_file = experiment_path / "inference_metrics.json"
    if metrics_file.exists():
        files["metrics"] = str(metrics_file)
    
    # Find results files
    for results_file in experiment_path.glob("results-*.json"):
        files["results"].append(str(results_file))
    
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Combine and analyze multiple experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python combine_results.py outputs/exp1 outputs/exp2 outputs/exp3
  python combine_results.py outputs/* --output-dir combined_results
  python combine_results.py outputs/exp1 outputs/exp2 --evaluate-only
  python combine_results.py outputs/exp1 outputs/exp2 --dataset gsm8k
        """
    )
    
    parser.add_argument('experiment_dirs', 
                       nargs='+',
                       help='Experiment directories containing inference_metrics.json and results files')
    parser.add_argument('--output-dir', 
                       default='combined_results',
                       help='Output directory for combined results (default: combined_results)')
    parser.add_argument('--dataset', 
                       help='Dataset name for evaluation (e.g., gsm8k, math)')
    parser.add_argument('--evaluate-only', 
                       action='store_true',
                       help='Only evaluate existing combined results, do not combine new ones')
    parser.add_argument('--dry-run', 
                       action='store_true',
                       help='Show what would be combined without actually combining')
    
    args = parser.parse_args()
    
    # Expand glob patterns
    experiment_dirs = []
    for pattern in args.experiment_dirs:
        if '*' in pattern or '?' in pattern:
            experiment_dirs.extend(glob.glob(pattern))
        else:
            experiment_dirs.append(pattern)
    
    if not experiment_dirs:
        print("Error: No experiment directories found")
        return 1
    
    print(f"Found {len(experiment_dirs)} experiment directories:")
    for dir_path in experiment_dirs:
        print(f"  {dir_path}")
    
    try:
        # Find all metrics and results files
        all_metrics = []
        all_results_files = []
        
        for exp_dir in experiment_dirs:
            try:
                files = find_experiment_files(exp_dir)
                if files["metrics"]:
                    all_metrics.append(files["metrics"])
                all_results_files.extend(files["results"])
                print(f"  {exp_dir}: {len(files['results'])} results files, {'metrics file' if files['metrics'] else 'no metrics'}")
            except Exception as e:
                print(f"Warning: Could not process {exp_dir}: {e}")
        
        if not all_metrics and not all_results_files:
            print("Error: No metrics or results files found in any experiment directory")
            return 1
        
        if args.dry_run:
            print(f"\nDRY RUN: Would combine:")
            print(f"  {len(all_metrics)} metrics files")
            print(f"  {len(all_results_files)} results files")
            return 0
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Combine metrics
        if all_metrics:
            print(f"\nCombining {len(all_metrics)} metrics files...")
            metrics_data = [load_inference_metrics(f) for f in all_metrics]
            combined_metrics = combine_inference_metrics(metrics_data)
            
            # Save combined metrics
            metrics_output = os.path.join(args.output_dir, "combined_inference_metrics.json")
            with open(metrics_output, 'w', encoding='utf-8') as f:
                json.dump(combined_metrics, f, indent=2, ensure_ascii=False)
            
            print(f"Combined metrics saved to: {metrics_output}")
            print(f"Total inference time: {combined_metrics['inference_time_seconds']:.3f} s")
            print(f"Total questions: {combined_metrics['total_questions']}")
            print(f"Average time per question: {combined_metrics['avg_time_per_question_seconds']:.3f} s")
            print(f"Peak memory: {combined_metrics['peak_memory_gb']:.2f} GB")
        
        # Combine results
        if all_results_files:
            print(f"\nCombining {len(all_results_files)} results files...")
            combined_results = combine_results_files(all_results_files)
            
            # Save combined results
            results_output = os.path.join(args.output_dir, "combined_results.json")
            with open(results_output, 'w', encoding='utf-8') as f:
                json.dump(combined_results, f, indent=2, ensure_ascii=False)
            
            print(f"Combined results saved to: {results_output}")
            print(f"Total samples: {len(combined_results)}")
            
            # Evaluate combined results
            print(f"\nEvaluating combined results...")
            evaluation = evaluate_results(combined_results, args.dataset)
            
            # Save evaluation results
            eval_output = os.path.join(args.output_dir, "evaluation_results.json")
            with open(eval_output, 'w', encoding='utf-8') as f:
                json.dump(evaluation, f, indent=2, ensure_ascii=False)
            
            print(f"Evaluation results saved to: {eval_output}")
            print(f"Accuracy: {evaluation['accuracy']:.4f}")
            print(f"Correct samples: {evaluation['correct_samples']}/{evaluation['total_samples']}")
            print(f"No match samples: {evaluation['no_match_samples']}")
        
        print(f"\nCombination complete! Results saved to: {args.output_dir}")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
