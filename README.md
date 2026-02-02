# MobilePupil

## Notice
Please note that our codebase makes extensive use of the implementation provided in the [FACET](https://github.com/DeanJY/FACET) repository. If you require further details regarding the underlying implementation or original methods, please refer to that repository directly.

## Dataset

For the creation of the binocular dataset, we provide several utility scripts to handle data preprocessing. You can refer to the following files for specific operations:

* **`alignment_slicing.py`**
    Aligns event data with ellipse labels. It identifies the nearest event timestamps corresponding to label entries and extracts event slices within a defined time window (e.g., 40ms) to generate paired data samples.

* **`downsampling.py`**
    Manages dataset size and balance. It performs proportional downsampling across all label files to reach a specific target total (e.g., 20,000 samples) while preserving the original temporal order and data distribution.

* **`ensure_time.py`**
    Ensures temporal consistency between event streams and labels. It iterates through data pairs and filters out invalid labels that lack sufficient preceding event history (i.e., labels occurring before the required time window duration).

> **Note:** The guide for the production process of the complete dataset or the prepared dataset will be uploaded later.


## 1. Train
To train the model, execute the `train.py` script with the corresponding configuration file:

```
python tools/train.py --config configs/DavisEyeEllipse_EPNet.yaml
```
## 2. Test

### Standard Evaluation
Use the following command to evaluate the trained model on the standard test set:

```bash
python EvEye/model/DavisEyeEllipse/EPNet/test.py
```

### Robustness Evaluation
To assess the robustness of the model against sensor noise and occlusions, we provide two specific testing scripts located in the same directory. These scripts simulate data degradation to verify model stability.

#### 1.`drop_event_test.py`
Simulates global event sparsity. 

* **Key Arguments:**
  * `--drop-rate`: The probability of dropping events (0.0 - 1.0).
  * `--batch-test`: Automatically iterates through drop rates from 0.1 to 0.9.
* **Output:** Generates visualization results comparing Ground Truth vs. Prediction under sparse conditions.

**Example:**

```bash
# Run a batch test
python drop_event_test.py --ckpt path/to/model.ckpt --data path/to/dataset --out path/to/save_dir --batch-test

# Run a single drop rate test
python drop_event_test.py --ckpt path/to/model.ckpt --data path/to/dataset --single-drop 0.5
```

#### 2.`mask_radio_test.py`
Simulates spatial occlusion by masking out random rectangular regions of the input event stream.

* **Key Arguments:**
  * `--mask-ratio`: The ratio of the image area to be occluded.
  * `--batch-test`: Automatically iterates through mask ratios from 0.1 to 0.9.
  * `--skip-eval` / `--skip-viz`: Flags to skip quantitative evaluation or visualization respectively.
* **Output:** Prints quantitative metrics and saves visualization images.

**Example:**

```bash
# Run a batch test with evaluation and visualization
python mask_radio_test.py --ckpt path/to/model.ckpt --data path/to/dataset --out path/to/save_dir --batch-test

# Run a specific occlusion ratio
python mask_radio_test.py --ckpt path/to/model.ckpt --data path/to/dataset --single-mask 0.3
```
