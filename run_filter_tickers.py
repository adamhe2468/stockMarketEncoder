"""
Script to filter ticker data files in the data folder.
Removes all ticker files where the ticker symbol has more than 5 letters.
"""

import os
import shutil
from pathlib import Path


def filter_ticker_files(data_dir: str, max_length: int = 4, backup: bool = True):
    """
    Filter ticker data files based on ticker symbol length and format.
    Only keeps tickers with 1-4 letters and no special characters (., -, etc.)

    Args:
        data_dir: Path to data directory containing ticker .txt files
        max_length: Maximum allowed ticker length (default: 4)
        backup: Whether to create backup before deletion (default: True)
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"Error: Directory not found: {data_dir}")
        return

    # Get all .txt files
    all_files = list(data_path.glob("*.txt"))

    if not all_files:
        print(f"No .txt files found in {data_dir}")
        return

    print(f"Found {len(all_files)} ticker files")
    print("=" * 60)

    # Create backup directory if requested
    if backup:
        backup_dir = data_path / "backup_filtered_tickers"
        backup_dir.mkdir(exist_ok=True)
        print(f"Backup directory: {backup_dir}")

    to_remove = []
    to_keep = []

    # Analyze files
    for file_path in all_files:
        # Extract ticker from filename (remove .txt extension)
        ticker = file_path.stem

        # Check if ticker is valid:
        # 1. Length must be 1-4 characters
        # 2. Must contain only letters (A-Z, a-z)
        # 3. No dots, hyphens, or other special characters
        is_valid = (
            1 <= len(ticker) <= max_length and
            ticker.isalpha()  # Only alphabetic characters
        )

        if not is_valid:
            to_remove.append((ticker, file_path))
        else:
            to_keep.append(ticker)

    print(f"\nTickers to KEEP (1-{max_length} letters, alphabetic only): {len(to_keep)}")
    print(f"Tickers to REMOVE (invalid format or length): {len(to_remove)}")
    print("=" * 60)

    if not to_remove:
        print("\nNo tickers to remove. All tickers are within length limit.")
        return

    # Show examples of what will be removed
    print(f"\nExamples of tickers being removed:")
    for ticker, _ in to_remove[:20]:
        print(f"  - {ticker} (length: {len(ticker)})")
    if len(to_remove) > 20:
        print(f"  ... and {len(to_remove) - 20} more")

    # Confirm deletion
    print("\n" + "=" * 60)
    response = input(f"Proceed with removing {len(to_remove)} ticker files? (yes/no): ")

    if response.lower() not in ['yes', 'y']:
        print("Operation cancelled.")
        return

    # Remove files
    removed_count = 0
    backed_up_count = 0

    for ticker, file_path in to_remove:
        try:
            # Backup if requested
            if backup:
                backup_path = backup_dir / file_path.name
                shutil.copy2(file_path, backup_path)
                backed_up_count += 1

            # Remove file
            file_path.unlink()
            removed_count += 1

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"Files removed: {removed_count}")
    if backup:
        print(f"Files backed up: {backed_up_count}")
    print(f"Files remaining: {len(to_keep)}")
    print(f"\nData folder now contains {len(to_keep)} ticker files")

    # Save list of removed tickers
    removed_list_path = data_path / "removed_tickers.txt"
    with open(removed_list_path, 'w') as f:
        f.write(f"# Tickers removed (invalid: length > {max_length} or contains special chars)\n")
        f.write(f"# Total: {len(to_remove)}\n\n")
        for ticker, _ in sorted(to_remove):
            reason = []
            if len(ticker) > max_length:
                reason.append(f"length={len(ticker)}")
            if not ticker.isalpha():
                reason.append("has special chars")
            f.write(f"{ticker}  # {', '.join(reason)}\n")

    print(f"\nList of removed tickers saved to: {removed_list_path}")


if __name__ == "__main__":
    import sys

    # Configuration
    DATA_DIR = "data"
    MAX_TICKER_LENGTH = 4
    CREATE_BACKUP = True

    print("=" * 60)
    print("Ticker Data File Filter")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR}")
    print(f"Max ticker length: {MAX_TICKER_LENGTH}")
    print(f"Only alphabetic characters (no dots, hyphens, etc.)")
    print(f"Create backup: {CREATE_BACKUP}")
    print("=" * 60)
    print()

    # Allow command line override
    if len(sys.argv) > 1:
        DATA_DIR = sys.argv[1]
    if len(sys.argv) > 2:
        MAX_TICKER_LENGTH = int(sys.argv[2])
    if len(sys.argv) > 3:
        CREATE_BACKUP = sys.argv[3].lower() in ['true', 'yes', '1']

    # Run filter
    filter_ticker_files(DATA_DIR, MAX_TICKER_LENGTH, CREATE_BACKUP)

    print("\nDone!")
