from pathlib import Path
import numpy as np
from natsort import natsorted
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class TxtProcessor:
    def __init__(self):
        pass

    def load_events_from_txt(self, txt_path):
        """Load event data, assuming format: t x y p (timestamp x-coordinate y-coordinate polarity)"""
        try:
            data = np.loadtxt(
                txt_path,
                dtype={'names': ('t', 'x', 'y', 'p'), 'formats': (np.float64, np.int32, np.int32, np.int32)},
                delimiter=' ',  # Use space delimiter
                comments='#'
            )
            return data
        except Exception as e:
            logging.error(f"Failed to load event data: {txt_path}, error: {e}")
            raise

    def load_labels_from_txt(self, txt_path):
        """Load label data, assuming format: t x y a b c (timestamp x-coordinate y-coordinate extra_params)"""
        try:
            # Specify reading only the first 4 columns (t, x, y, a)
            data = np.loadtxt(
                txt_path,
                dtype={'names': ('t', 'x', 'y', 'a'), 'formats': (np.float64, np.float64, np.float64, np.float64)},
                delimiter=' ',
                comments='#',
                usecols=(0, 1, 2, 3)  # Read only first 4 columns
            )

            # Create new structured array, handling data types correctly
            new_data = np.zeros(len(data),
                                dtype=[('t', np.float64), ('x', np.float64), ('y', np.float64), ('close', np.int32)])
            new_data['t'] = data['t']
            new_data['x'] = data['x']
            new_data['y'] = data['y']

            # Handle the fourth column appropriately
            # Assuming float needs to be converted to int (may need rounding)
            new_data['close'] = np.round(data['a']).astype(np.int32)

            return new_data
        except Exception as e:
            logging.error(f"Failed to load label data: {txt_path}, error: {e}")
            self._diagnose_file_format(txt_path)
            raise

    def _diagnose_file_format(self, txt_path):
        """Read the first few lines of the file to help diagnose format issues"""
        try:
            with open(txt_path, 'r') as f:
                lines = [f.readline().strip() for _ in range(3)]
                logging.info(f"First few lines of file {txt_path}:")
                for line in lines:
                    logging.info(f"  {line}")
        except Exception as e:
            logging.error(f"Failed to diagnose file format: {e}")


def save_labels_to_txt(labels, txt_path):
    """Save label data to space-delimited txt file, keeping the first 4 columns"""
    try:
        with open(txt_path, 'w') as f:
            for i in range(len(labels)):
                # Save four columns: t x y close
                label_line = f"{labels[i]['t']} {labels[i]['x']} {labels[i]['y']} {labels[i]['close']}\n"
                f.write(label_line)
        logging.info(f"Successfully saved labels to: {txt_path}")
    except Exception as e:
        logging.error(f"Failed to save labels: {txt_path}, error: {e}")
        raise


def ensure_labels(root_path):
    """Ensure temporal consistency between label data and event data"""
    data_base_path = root_path / "data"
    label_base_path = root_path / "label"

    # Check if directories exist
    if not data_base_path.exists():
        logging.error(f"Data directory does not exist: {data_base_path}")
        return

    if not label_base_path.exists():
        logging.error(f"Label directory does not exist: {label_base_path}")
        return

    data_paths = natsorted(data_base_path.glob("*.txt"))
    label_paths = natsorted(label_base_path.glob("*.txt"))

    # Check if file counts match
    if len(data_paths) != len(label_paths):
        logging.warning(f"Mismatch in data and label file counts: {len(data_paths)} vs {len(label_paths)}")

    time_window = 40000  # Time window size

    for index, (data_path, label_path) in enumerate(
            tqdm(zip(data_paths, label_paths), total=min(len(data_paths), len(label_paths)))
    ):
        logging.info(f"Processing file pair: {data_path.name} <-> {label_path.name}")

        try:
            # Load data
            processor = TxtProcessor()
            event = processor.load_events_from_txt(data_path)
            label = processor.load_labels_from_txt(label_path)

            if len(event) == 0:
                logging.warning(f"Event file is empty: {data_path}")
                continue

            if len(label) == 0:
                logging.warning(f"Label file is empty: {label_path}")
                continue

            # Find the first valid label
            index_label = 0
            while index_label < len(label):
                start_time_first = max(event['t'][0], label['t'][index_label] - time_window)
                end_time_first = label['t'][index_label]

                if start_time_first >= end_time_first:
                    logging.warning(f"Invalid label time: {label['t'][index_label]}, file: {label_path}")
                    index_label += 1
                    if index_label < len(label):
                        continue
                    else:
                        raise ValueError(f"No valid labels in file {label_path}")
                else:
                    break

            # Filter labels
            if index_label > 0:
                original_count = len(label)
                label = label[index_label:]
                logging.info(
                    f"Filtered out {index_label} invalid labels from {label_path}, keeping {len(label)}/{original_count}")

            # Save processed labels
            save_labels_to_txt(label, label_path)

        except Exception as e:
            logging.error(f"Failed to process file pair: {data_path.name} <-> {label_path.name}, error: {e}")
            # Continue processing other file pairs
            continue


def main():
    # Process training set
    train_path = Path("D:/A/FACET-main/ensure_test")
    logging.info(f"Starting processing of training set: {train_path}")
    ensure_labels(train_path)


if __name__ == "__main__":
    main()