#!/usr/bin/env python3
"""
Cleanup Script for Temporary Partitions

This script removes the temporary partition files created by partition_dataset.py.
It can clean up specific directories or use default locations.

Usage:
    python cleanup_partitions.py [--dir DIRECTORY] [--dry-run]

Examples:
    python cleanup_partitions.py
    python cleanup_partitions.py --dir ../benchmark/my_partitions
    python cleanup_partitions.py --dry-run  # Preview what would be deleted
"""

import argparse
import os
import shutil
from pathlib import Path


def cleanup_directory(directory_path, dry_run=False):
    """Remove all files and subdirectories in the given directory."""
    directory_path = Path(directory_path)
    
    if not directory_path.exists():
        print(f"Directory does not exist: {directory_path}")
        return False
    
    if not directory_path.is_dir():
        print(f"Path is not a directory: {directory_path}")
        return False
    
    # Count files before deletion
    files_count = 0
    total_size = 0
    
    for root, dirs, files in os.walk(directory_path):
        files_count += len(files)
        for file in files:
            file_path = os.path.join(root, file)
            try:
                total_size += os.path.getsize(file_path)
            except OSError:
                pass  # Skip files we can't access
    
    if files_count == 0:
        print(f"Directory is already empty: {directory_path}")
        return True
    
    # Show what will be deleted
    print(f"Found {files_count} files ({total_size / 1024:.1f} KB) in {directory_path}")
    
    if dry_run:
        print("DRY RUN: Would delete the following files:")
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                print(f"  {file_path}")
        return True
    
    # Confirm deletion
    try:
        shutil.rmtree(directory_path)
        print(f"Successfully deleted directory: {directory_path}")
        print(f"Removed {files_count} files ({total_size / 1024:.1f} KB)")
        return True
    except Exception as e:
        print(f"Error deleting directory {directory_path}: {e}")
        return False


def cleanup_partition_files(directory_path, dry_run=False):
    """Remove only partition files, keeping the directory structure."""
    directory_path = Path(directory_path)
    
    if not directory_path.exists():
        print(f"Directory does not exist: {directory_path}")
        return False
    
    if not directory_path.is_dir():
        print(f"Path is not a directory: {directory_path}")
        return False
    
    # Find partition files
    partition_files = []
    for file_path in directory_path.rglob("*_partition_*.json"):
        partition_files.append(file_path)
    
    if not partition_files:
        print(f"No partition files found in: {directory_path}")
        return True
    
    # Show what will be deleted
    total_size = sum(f.stat().st_size for f in partition_files)
    print(f"Found {len(partition_files)} partition files ({total_size / 1024:.1f} KB)")
    
    if dry_run:
        print("DRY RUN: Would delete the following files:")
        for file_path in partition_files:
            print(f"  {file_path}")
        return True
    
    # Delete files
    deleted_count = 0
    for file_path in partition_files:
        try:
            file_path.unlink()
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    print(f"Successfully deleted {deleted_count}/{len(partition_files)} partition files")
    return deleted_count == len(partition_files)


def main():
    parser = argparse.ArgumentParser(
        description="Clean up temporary partition files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cleanup_partitions.py                           # Clean default directory
  python cleanup_partitions.py --dir ../benchmark/my_partitions  # Clean specific directory
  python cleanup_partitions.py --dry-run                 # Preview what would be deleted
  python cleanup_partitions.py --files-only              # Delete only partition files, keep directory
        """
    )
    
    parser.add_argument('--dir', 
                       default='../benchmark/temp_partitions',
                       help='Directory to clean up (default: ../benchmark/temp_partitions)')
    parser.add_argument('--dry-run', 
                       action='store_true',
                       help='Show what would be deleted without actually deleting')
    parser.add_argument('--files-only', 
                       action='store_true',
                       help='Delete only partition files, keep the directory structure')
    
    args = parser.parse_args()
    
    # Convert relative path to absolute
    directory_path = os.path.abspath(args.dir)
    
    print(f"Cleaning up: {directory_path}")
    
    try:
        if args.files_only:
            success = cleanup_partition_files(directory_path, args.dry_run)
        else:
            success = cleanup_directory(directory_path, args.dry_run)
        
        if success:
            print("Cleanup completed successfully!")
            return 0
        else:
            print("Cleanup failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\nCleanup cancelled by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
