import os
import random
import argparse
from typing import List, Tuple, Dict


def read_ellipse_data(file_path: str) -> List[str]:
    """Read ellipse data file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.readlines()
    except Exception as e:
        print(f"Failed to read file {file_path}: {e}")
        return []


def sample_ellipse_data(data_lines: List[str], target_lines: int) -> List[str]:
    """
    Sample data from a single file, maintaining original order

    Args:
        data_lines: All lines in the file
        target_lines: Number of lines to keep for the current file

    Returns:
        List of sampled lines (maintaining original order)
    """
    if not data_lines:
        return []

    total_lines = len(data_lines)

    # If file lines are less than target lines, keep all
    if total_lines <= target_lines:
        print(f"Warning: File line count ({total_lines}) is less than target count ({target_lines}), keeping all lines")
        return data_lines

    # Calculate sampling interval to maintain uniform distribution
    step = total_lines / target_lines
    indices = [int(i * step) for i in range(target_lines)]

    # Select lines in original order
    return [data_lines[i] for i in indices]


def write_sampled_data(file_path: str, data_lines: List[str]) -> None:
    """Write sampled data"""
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(data_lines)
        print(f"Sampled data written to {file_path}")
    except Exception as e:
        print(f"Failed to write file {file_path}: {e}")


def main(root_path: str, total_target_lines: int = 20000) -> None:
    """Main function: Process all ellipse data files in the specified directory"""
    # Get all txt files
    txt_files = [f for f in os.listdir(root_path) if f.endswith('.txt')]
    if not txt_files:
        print(f"No txt files found in directory {root_path}")
        return

    # Read line counts of all files
    file_line_counts: Dict[str, int] = {}
    total_lines = 0

    for txt_file in txt_files:
        file_path = os.path.join(root_path, txt_file)
        lines = read_ellipse_data(file_path)
        count = len(lines)
        file_line_counts[txt_file] = (count, lines)
        total_lines += count

    print(f"Found {len(txt_files)} txt files, total lines: {total_lines}")
    print(f"Total target lines: {total_target_lines}")

    if total_lines <= total_target_lines:
        print("Total data lines already less than or equal to target, no sampling needed")
        return

    # Calculate lines to keep for each file - use proportional allocation to ensure exact total
    file_target_counts: Dict[str, int] = {}

    # Allocate target lines based on original line count proportion
    remaining_lines = total_target_lines
    remaining_files = len(txt_files)

    # Sort by line count, process files with more lines first
    sorted_files = sorted(file_line_counts.items(),
                          key=lambda x: x[1][0], reverse=True)

    # Calculate lines to keep per file (proportional)
    for file_name, (count, _) in sorted_files:
        if remaining_files <= 0:
            break

        # Calculate theoretical lines to keep (proportional)
        if total_lines > 0:
            theoretical_lines = int(count * total_target_lines / total_lines)
        else:
            theoretical_lines = 0

        # Ensure at least 1 line is kept (if possible)
        min_lines = min(count, max(1, theoretical_lines))

        # Process the last file to ensure exact total
        if remaining_files == 1:
            target_lines = remaining_lines
        else:
            # Avoid allocating too many lines resulting in exceeding target
            target_lines = min(min_lines, remaining_lines - (remaining_files - 1))

        file_target_counts[file_name] = target_lines
        remaining_lines -= target_lines
        remaining_files -= 1

    # Execute sampling and write to files
    current_total = 0
    for file_name, (count, lines) in sorted_files:
        target_lines = file_target_counts.get(file_name, 0)
        if target_lines <= 0:
            print(f"Warning: File {file_name} was not allocated any lines to keep")
            continue

        sampled_lines = sample_ellipse_data(lines, target_lines)
        current_total += len(sampled_lines)

        file_path = os.path.join(root_path, file_name)
        write_sampled_data(file_path, sampled_lines)

    print(f"Sampling complete, final total lines: {current_total} (Target: {total_target_lines})")
    if current_total != total_target_lines:
        print(f"Warning: Final line count differs from target, difference: {current_total - total_target_lines}")
        print("This may be due to some files having too few lines")


if __name__ == "__main__":
    # Set command line arguments
    parser = argparse.ArgumentParser(description='Ellipse data sampling tool')
    parser.add_argument('--root_path', type=str, required=True,
                        help='Directory containing ellipse data files')
    parser.add_argument('--nums', type=int, default=20000,
                        help='Total number of lines to keep')
    args = parser.parse_args()

    main(args.root_path, args.nums)