import os
from collections import defaultdict


def find_nearest_event(events, ellipse_time):
    """Find the event timestamp nearest to the ellipse timestamp"""
    left, right = 0, len(events) - 1
    nearest_idx = 0
    min_diff = float('inf')

    while left <= right:
        mid = (left + right) // 2
        diff = abs(events[mid][0] - ellipse_time)

        if diff < min_diff:
            min_diff = diff
            nearest_idx = mid

        if events[mid][0] < ellipse_time:
            left = mid + 1
        elif events[mid][0] > ellipse_time:
            right = mid - 1
        else:
            break

    return nearest_idx


def align_event_ellipse(event_path, ellipse_path, save_dir, timewindow=5000):
    """Align event data with ellipse data based on timestamps"""
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Read event data
    print(f"Reading event data: {event_path}")
    with open(event_path, 'r') as f:
        event_lines = [line.strip() for line in f.readlines()]

    # Parse event data and sort by timestamp
    events = []
    for line in event_lines:
        parts = line.split()
        if len(parts) >= 4:
            try:
                t = int(parts[0])
                x = int(parts[1])
                y = int(parts[2])
                p = int(parts[3])
                events.append((t, x, y, p, line))
            except ValueError:
                continue

    if not events:
        print("No valid event data found!")
        return

    events.sort(key=lambda e: e[0])
    event_times = [e[0] for e in events]
    print(f"Read {len(events)} event data entries")

    # Read ellipse data
    print(f"Reading ellipse data: {ellipse_path}")
    with open(ellipse_path, 'r') as f:
        ellipse_lines = [line.strip() for line in f.readlines()]

    # Parse ellipse data
    valid_ellipse_lines = []
    for line in ellipse_lines:
        parts = line.split()
        if len(parts) >= 6:
            try:
                t = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                a = float(parts[3])
                b = float(parts[4])
                theta = (float(parts[5]))
                valid_ellipse_lines.append((t, x, y, a, b, theta, line))
            except ValueError:
                continue

    if not valid_ellipse_lines:
        print("No valid ellipse data found!")
        return

    print(f"Read {len(valid_ellipse_lines)} ellipse data entries")

    # Process each ellipse data entry
    print("Processing data alignment...")
    output_ellipse_lines = []
    processed_count = 0

    for ellipse in valid_ellipse_lines:
        ellipse_time = ellipse[0]
        nearest_idx = find_nearest_event(events, ellipse_time)

        # Determine the range of events within the time window
        start_idx = max(0, nearest_idx - timewindow + 1)
        end_idx = nearest_idx + 1

        # Check if the number of events is sufficient
        if end_idx - start_idx >= timewindow:
            # Extract event data
            selected_events = events[start_idx:end_idx]

            # Generate output filename
            file_idx = processed_count + 1
            output_filename = os.path.join(save_dir, f"event_{file_idx}.txt")

            # Write event data
            with open(output_filename, 'w') as f:
                for event in selected_events:
                    f.write(event[4] + '\n')

            # Record valid ellipse line
            output_ellipse_lines.append(ellipse[6])
            processed_count += 1

    # Save filtered ellipse data
    output_ellipse_path = os.path.join(save_dir, "ellipse_filtered.txt")
    with open(output_ellipse_path, 'w') as f:
        for line in output_ellipse_lines:
            f.write(line + '\n')

    print(f"Processing complete! Generated {processed_count} event files")
    print(f"Filtered ellipse data saved to: {output_ellipse_path}")


if __name__ == "__main__":
    # Configuration parameters
    event_path = "D:/A/FACET-main/ensure_test/data/1.txt"  # Path to event data file
    ellipse_path = "D:/A/FACET-main/ensure_test/label/1.txt"  # Path to ellipse data file
    save_dir = "D:/A/FACET-main/ensure_test/40000_results"  # Directory to save results
    timewindow = 40000  # Time window size

    # Execute alignment operation
    align_event_ellipse(event_path, ellipse_path, save_dir, timewindow)