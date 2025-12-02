# Rule-Based Scenario Classifier

This directory contains the rule-based scenario classifier and associated tools.

## Threshold Tuning (Unsupervised Learning)

The `tune_thresholds.py` script automatically tunes the classification thresholds using unsupervised learning (Gaussian Mixture Models) on CAN bus data.

### Usage

Run the script using `uv`:

**Basic Usage (v1.0-mini)**

```bash
uv run python tune_thresholds.py
```

**Full Dataset (v1.0-trainval)**

To use the full dataset, specify the version:

```bash
uv run python tune_thresholds.py --version v1.0-trainval
```

**Custom Data Root**

If your NuScenes data is not in the default location (`../../data/nuscenes`), specify the path:

```bash
uv run python tune_thresholds.py --version v1.0-trainval --dataroot /path/to/your/nuscenes
```

### Output

The script generates:
- `config_suggested.yaml`: A configuration file with the suggested thresholds.
- Distribution Plots:
    - `dist_speed.png`
    - `dist_steering.png`
    - `dist_yaw.png`

Review the plots and the suggested config before updating `config.yaml`.
