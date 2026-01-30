#!/usr/bin/env python3
"""
GNHK Dataset Downloader
Downloads the GoodNotes Handwriting Kollection from Google Drive

Usage:
    python download_gnhk.py --output ../datasets/gnhk
    python download_gnhk.py --format sagemaker  # or 'paper'
"""

import os
import sys
import argparse
import zipfile
import shutil
from pathlib import Path

try:
    import gdown
except ImportError:
    print("Installing gdown...")
    os.system(f"{sys.executable} -m pip install gdown")
    import gdown


# GNHK Google Drive folder IDs
GNHK_URLS = {
    "sagemaker": {
        "folder_id": "1KXu55SBzyyZf0Ek-F6Kudre1tWd9dDC2",
        "url": "https://drive.google.com/drive/folders/1KXu55SBzyyZf0Ek-F6Kudre1tWd9dDC2",
        "description": "SageMaker JSON format (recommended for TrOCR)"
    },
    "paper": {
        "folder_id": "1YMdE1cP9ZSUxzwycCbrgQjVN_cy2gIw1",
        "url": "https://drive.google.com/drive/folders/1YMdE1cP9ZSUxzwycCbrgQjVN_cy2gIw1",
        "description": "Paper JSON format"
    }
}


def download_gnhk(output_dir: str, format_type: str = "sagemaker", quiet: bool = False):
    """
    Download the GNHK dataset from Google Drive.

    Args:
        output_dir: Directory to save the dataset
        format_type: 'sagemaker' or 'paper'
        quiet: Suppress progress output
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if format_type not in GNHK_URLS:
        raise ValueError(f"Unknown format: {format_type}. Use 'sagemaker' or 'paper'")

    config = GNHK_URLS[format_type]

    print(f"=" * 60)
    print(f"GNHK Dataset Downloader")
    print(f"=" * 60)
    print(f"Format: {format_type} - {config['description']}")
    print(f"Output: {output_path.absolute()}")
    print(f"Source: {config['url']}")
    print(f"=" * 60)
    print()

    # Download entire folder
    print(f"Downloading GNHK dataset ({format_type} format)...")
    print("Note: This may take several minutes depending on your connection.")
    print()

    try:
        gdown.download_folder(
            url=config['url'],
            output=str(output_path),
            quiet=quiet,
            use_cookies=False
        )
        print()
        print(f"Download complete!")

        # List downloaded files
        print(f"\nDownloaded files:")
        for f in output_path.rglob("*"):
            if f.is_file():
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  {f.relative_to(output_path)} ({size_mb:.2f} MB)")

        return True

    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("\nManual download instructions:")
        print(f"  1. Go to: {config['url']}")
        print(f"  2. Download all files")
        print(f"  3. Extract to: {output_path.absolute()}")
        return False


def verify_dataset(dataset_dir: str) -> dict:
    """
    Verify the downloaded dataset structure.

    Returns dict with dataset statistics.
    """
    dataset_path = Path(dataset_dir)

    stats = {
        "valid": False,
        "train_images": 0,
        "test_images": 0,
        "annotations": [],
        "total_size_mb": 0
    }

    if not dataset_path.exists():
        print(f"Dataset directory not found: {dataset_path}")
        return stats

    # Count images
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        train_imgs = list(dataset_path.glob(f"**/train*/{ext}"))
        test_imgs = list(dataset_path.glob(f"**/test*/{ext}"))
        stats["train_images"] += len(train_imgs)
        stats["test_images"] += len(test_imgs)

    # Find annotation files
    for pattern in ['*.json', '*.manifest']:
        stats["annotations"].extend([str(f) for f in dataset_path.rglob(pattern)])

    # Calculate total size
    total_size = sum(f.stat().st_size for f in dataset_path.rglob("*") if f.is_file())
    stats["total_size_mb"] = total_size / (1024 * 1024)

    stats["valid"] = stats["train_images"] > 0 or len(stats["annotations"]) > 0

    print(f"\nDataset Statistics:")
    print(f"  Training images: {stats['train_images']}")
    print(f"  Test images: {stats['test_images']}")
    print(f"  Annotation files: {len(stats['annotations'])}")
    print(f"  Total size: {stats['total_size_mb']:.2f} MB")
    print(f"  Valid: {'Yes' if stats['valid'] else 'No'}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download GNHK Handwriting Dataset from Google Drive"
    )
    parser.add_argument(
        "--output", "-o",
        default="./datasets/gnhk",
        help="Output directory for dataset (default: ./datasets/gnhk)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["sagemaker", "paper"],
        default="sagemaker",
        help="Dataset format to download (default: sagemaker)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing dataset, don't download"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress download progress"
    )

    args = parser.parse_args()

    if args.verify_only:
        verify_dataset(args.output)
    else:
        success = download_gnhk(args.output, args.format, args.quiet)
        if success:
            verify_dataset(args.output)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
