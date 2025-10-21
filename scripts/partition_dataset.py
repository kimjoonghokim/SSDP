#!/usr/bin/env python3
"""
Dataset Partitioning Script

This script takes a dataset file and partitions it into multiple smaller files.
Each partition will contain roughly equal number of samples.

Usage:
    python partition_dataset.py <dataset_path> <num_partitions> [--output-dir OUTPUT_DIR]

Example:
    python partition_dataset.py ../benchmark/gsm8k_toy20_dataset.json 4
    python partition_dataset.py ../benchmark/math_test500_dataset.json 10 --output-dir ../benchmark/temp_partitions
"""

import argparse
import json
import os
import math
from pathlib import Path


def load_dataset(dataset_path):
    """Load dataset from JSON file."""
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in dataset file: {e}")


def partition_dataset(data, num_partitions):
    """Partition dataset into roughly equal parts."""
    if not data:
        raise ValueError("Dataset is empty")
    
    if num_partitions <= 0:
        raise ValueError("Number of partitions must be positive")
    
    if num_partitions > len(data):
        print(f"Warning: Number of partitions ({num_partitions}) is greater than dataset size ({len(data)}). "
              f"Some partitions will be empty.")
    
    # Calculate partition sizes
    total_samples = len(data)
    base_size = total_samples // num_partitions
    remainder = total_samples % num_partitions
    
    partitions = []
    start_idx = 0
    
    for i in range(num_partitions):
        # Add one extra sample to the first 'remainder' partitions
        partition_size = base_size + (1 if i < remainder else 0)
        end_idx = start_idx + partition_size
        
        partition_data = data[start_idx:end_idx]
        partitions.append(partition_data)
        
        start_idx = end_idx
    
    return partitions


def save_partitions(partitions, output_dir, base_filename):
    """Save partitions to separate JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    base_name = Path(base_filename).stem  # Remove .json extension
    
    for i, partition in enumerate(partitions):
        if not partition:  # Skip empty partitions
            continue
            
        filename = f"{base_name}_partition_{i+1:03d}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(partition, f, indent=2, ensure_ascii=False)
        
        saved_files.append(filepath)
        print(f"Saved partition {i+1}: {filename} ({len(partition)} samples)")
    
    return saved_files


def main():
    parser = argparse.ArgumentParser(
        description="Partition a dataset into multiple smaller files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python partition_dataset.py ../benchmark/gsm8k_toy20_dataset.json 4
  python partition_dataset.py ../benchmark/math_test500_dataset.json 10 --output-dir ../benchmark/temp_partitions
        """
    )
    
    parser.add_argument('dataset_path', 
                       help='Path to the dataset JSON file')
    parser.add_argument('num_partitions', 
                       type=int,
                       help='Number of partitions to create')
    parser.add_argument('--output-dir', 
                       default='../benchmark/temp_partitions',
                       help='Output directory for partitions (default: ../benchmark/temp_partitions)')
    
    args = parser.parse_args()
    
    try:
        # Load dataset
        print(f"Loading dataset from: {args.dataset_path}")
        data = load_dataset(args.dataset_path)
        print(f"Loaded {len(data)} samples")
        
        # Partition dataset
        print(f"Partitioning into {args.num_partitions} parts...")
        partitions = partition_dataset(data, args.num_partitions)
        
        # Save partitions
        print(f"Saving partitions to: {args.output_dir}")
        saved_files = save_partitions(partitions, args.output_dir, args.dataset_path)
        
        # Summary
        print(f"\nPartitioning complete!")
        print(f"Created {len(saved_files)} partition files")
        print(f"Total samples: {len(data)}")
        print(f"Average samples per partition: {len(data) / args.num_partitions:.1f}")
        
        # Show partition sizes
        print("\nPartition sizes:")
        for i, partition in enumerate(partitions):
            if partition:  # Only show non-empty partitions
                print(f"  Partition {i+1}: {len(partition)} samples")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
