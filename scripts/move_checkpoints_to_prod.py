#!/usr/bin/env python3
"""
Script to move the latest trained model checkpoints to production folder.

This script automatically finds the most recent training runs and moves the best.ckpt
files to the production models folder while maintaining the correct folder structure.
It handles both single runs and multiruns.

Usage:
    python scripts/move_checkpoints_to_prod.py [--dry-run] [--force] [--run-type {runs,multiruns,both}]

Options:
    --dry-run: Show what would be moved without actually moving files
    --force: Overwrite existing checkpoints in production folder
    --run-type: Type of training runs to search for (runs, multiruns, or both)
"""

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def find_latest_run_or_multirun(
    logs_dir: Path, run_type: str = "both"
) -> Optional[Path]:
    """
    Find the most recent training run or multirun directory.

    Args:
        logs_dir: Path to the logs/train directory
        run_type: Type of runs to search for. Options: "runs", "multiruns", "both"

    Returns:
        Path to the most recent run/multirun directory, or None if not found
    """
    runs_dir = logs_dir / "runs"
    multiruns_dir = logs_dir / "multiruns"

    latest_timestamp = None
    latest_path = None

    # Check runs directory
    if run_type in ["runs", "both"] and runs_dir.exists():
        for run_dir in runs_dir.iterdir():
            if run_dir.is_dir() and run_dir.name.startswith("20"):
                try:
                    timestamp = datetime.strptime(run_dir.name, "%Y-%m-%d_%H-%M-%S")
                    if latest_timestamp is None or timestamp > latest_timestamp:
                        latest_timestamp = timestamp
                        latest_path = run_dir
                except ValueError:
                    continue

    # Check multiruns directory
    if run_type in ["multiruns", "both"] and multiruns_dir.exists():
        for multirun_dir in multiruns_dir.iterdir():
            if multirun_dir.is_dir() and multirun_dir.name.startswith("20"):
                try:
                    timestamp = datetime.strptime(
                        multirun_dir.name, "%Y-%m-%d_%H-%M-%S"
                    )
                    if latest_timestamp is None or timestamp > latest_timestamp:
                        latest_timestamp = timestamp
                        latest_path = multirun_dir
                except ValueError:
                    continue

    return latest_path


def find_species_checkpoints(latest_run: Path) -> Dict[str, Path]:
    """
    Find all species checkpoint directories in the latest run.

    Args:
        latest_run: Path to the latest run or multirun directory

    Returns:
        Dictionary mapping species names to their checkpoint directories
    """
    species_checkpoints = {}

    for species_dir in latest_run.iterdir():
        if species_dir.is_dir() and not species_dir.name.startswith("."):
            # Skip multirun.yaml and other non-species files
            if species_dir.name == "multirun.yaml":
                continue

            checkpoint_dir = species_dir / "checkpoints"
            best_checkpoint = checkpoint_dir / "best.ckpt"

            if best_checkpoint.exists():
                species_checkpoints[species_dir.name] = checkpoint_dir
            else:
                print(f"Warning: No best.ckpt found for {species_dir.name}")

    return species_checkpoints


def move_checkpoint_to_prod(
    species_name: str,
    source_checkpoint_dir: Path,
    prod_models_dir: Path,
    dry_run: bool = False,
    force: bool = False,
) -> bool:
    """
    Move a species checkpoint to the production folder.

    Args:
        species_name: Name of the species
        source_checkpoint_dir: Source checkpoint directory
        prod_models_dir: Production models directory
        dry_run: If True, only show what would be done
        force: If True, overwrite existing files

    Returns:
        True if successful (or would be successful in dry-run), False otherwise
    """
    source_best = source_checkpoint_dir / "best.ckpt"
    source_last = source_checkpoint_dir / "last.ckpt"

    target_species_dir = prod_models_dir / species_name
    target_checkpoint_dir = target_species_dir / "checkpoints"
    target_best = target_checkpoint_dir / "best.ckpt"
    target_last = target_checkpoint_dir / "last.ckpt"

    if not source_best.exists():
        print(f"Error: Source best.ckpt not found for {species_name}")
        return False

    # Create target directory if it doesn't exist
    if not dry_run:
        target_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Check if target exists and handle overwrite logic
    if target_best.exists() and not force:
        print(f"Warning: {target_best} already exists. Use --force to overwrite.")
        return False

    if dry_run:
        print(f"[DRY RUN] Would copy {source_best} -> {target_best}")
        if source_last.exists():
            print(f"[DRY RUN] Would copy {source_last} -> {target_last}")
    else:
        try:
            # Copy best.ckpt
            shutil.copy2(source_best, target_best)
            print(f"Copied {source_best} -> {target_best}")

            # Copy last.ckpt if it exists
            if source_last.exists():
                shutil.copy2(source_last, target_last)
                print(f"Copied {source_last} -> {target_last}")

            print(f"✓ Successfully updated checkpoints for {species_name}")

        except Exception as e:
            print(f"Error copying checkpoints for {species_name}: {e}")
            return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Move latest trained model checkpoints to production folder"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be moved without actually moving files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing checkpoints in production folder",
    )
    parser.add_argument(
        "--run-type",
        choices=["runs", "multiruns", "both"],
        default="both",
        help="Type of training runs to search for (default: both)",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs/train"),
        help="Path to training logs directory (default: logs/train)",
    )
    parser.add_argument(
        "--prod-dir",
        type=Path,
        default=Path("prod/models"),
        help="Path to production models directory (default: prod/models)",
    )

    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent.parent  # Go up one level from scripts/
    logs_dir = script_dir / args.logs_dir
    prod_models_dir = script_dir / args.prod_dir

    print(f"Looking for latest training runs in: {logs_dir}")
    print(f"Run type filter: {args.run_type}")
    print(f"Production models directory: {prod_models_dir}")

    if not logs_dir.exists():
        print(f"Error: Logs directory {logs_dir} does not exist")
        return 1

    if not prod_models_dir.exists():
        print(f"Error: Production models directory {prod_models_dir} does not exist")
        return 1

    # Find the latest run
    latest_run = find_latest_run_or_multirun(logs_dir, args.run_type)
    if latest_run is None:
        print(f"No training runs found for run type: {args.run_type}")
        return 1

    print(f"Found latest training run: {latest_run}")

    # Find species checkpoints
    species_checkpoints = find_species_checkpoints(latest_run)
    if not species_checkpoints:
        print("No species checkpoints found in the latest run")
        return 1

    print(f"Found checkpoints for {len(species_checkpoints)} species:")
    for species in species_checkpoints.keys():
        print(f"  - {species}")

    if args.dry_run:
        print("\n--- DRY RUN MODE ---")

    # Move checkpoints
    successful = 0
    failed = 0

    for species_name, checkpoint_dir in species_checkpoints.items():
        success = move_checkpoint_to_prod(
            species_name,
            checkpoint_dir,
            prod_models_dir,
            dry_run=args.dry_run,
            force=args.force,
        )

        if success:
            successful += 1
        else:
            failed += 1

    # Summary
    print(f"\n--- Summary ---")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")

    if args.dry_run:
        print("This was a dry run. Use without --dry-run to actually move files.")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
