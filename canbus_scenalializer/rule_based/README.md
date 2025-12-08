# Rule-Based Scenario Classifier

This directory contains the rule-based scenario classifier and associated tools.

## Prerequisites

Before running any scripts, ensure you have the necessary data and environment set up.

### 1. Data Setup

The scripts expect the NuScenes dataset to be located at `../../data/nuscenes` (relative to this directory).
Specifically, you need the **CAN bus expansion** data.

**Directory Structure:**

```text
project_root/
  ├── data/
  │   └── nuscenes/
  │       ├── can_bus/          <-- Required: Contains scene-*.json files
  │       ├── v1.0-mini/        <-- Required for demo/testing
  │       ├── maps/             <-- Required for NuScenes initialization
  │       └── ...
  └── canbus_scenalializer/
      └── rule_based/           <-- You are here
```

If your data is located elsewhere, you may need to modify the `dataroot` variable in the scripts or pass it as an argument (where supported, e.g., `tune_thresholds.py`).

### 2. Environment Setup

This project uses `uv` for dependency management. Ensure `uv` is installed.

```bash
# Install dependencies (if not already done)
uv sync
```

All scripts should be run using `uv run` to ensure they use the correct virtual environment and dependencies.

## Configuration

The classification logic is controlled by `config.yaml`. This file defines the thresholds used to categorize vehicle behavior.

**`config.yaml` Parameters:**

- `stop_speed_threshold`: Speed (m/s) below which the vehicle is considered "Stopped".
- `pull_over_speed_threshold`: Speed (m/s) below which (but above stop) the vehicle might be "Pulling Over".
- `lane_change_steering_threshold`: Steering angle (rad) threshold for detecting Lane Changes.
- `turn_steering_threshold`: Steering angle (rad) threshold for detecting Turns.
- `u_turn_steering_threshold`: Steering angle (rad) threshold for detecting U-Turns.
- `yaw_rate_threshold`: Yaw rate (rad/s) threshold for confirming turns.
- `turn_signal_on_threshold`: Threshold for binary turn signal state (if analog).

You can manually edit this file or use `tune_thresholds.py` to generate suggested values.

## Scripts & Usage

### 1. Generate Demo Scenes (`generate_demo_scenes.py`)

This script generates visualization videos and JSON classification results for the first 10 scenes of the NuScenes dataset. It is useful for verifying the classifier's performance on a batch of data.

**Usage:**

```bash
uv run python generate_demo_scenes.py
```

**Options:**
- This script currently has no command-line arguments.
- **Data Path**: It assumes NuScenes data is located at `../../data/nuscenes`.
- **Scene Count**: Hardcoded to process the first 10 scenes.

**Output:**
- Results are saved in `../output/{timestamp}/`.
- For each scene (e.g., `scene-0061`), it creates:
    - `demo_output.mp4`: A video with the scenario name overlayed.
    - `classification_results.json`: Detailed classification results for each timestamp.

### 2. Threshold Tuning (`tune_thresholds.py`)

Automatically tunes the classification thresholds using Unsupervised Learning (Gaussian Mixture Models) on the CAN bus data.

**Usage:**

```bash
uv run python tune_thresholds.py [OPTIONS]
```

**Options:**

- `--dataroot`: Path to the NuScenes data root.
    - Default: `../../data/nuscenes`
- `--version`: NuScenes version to use.
    - Default: `v1.0-mini`
    - Example: `--version v1.0-trainval`

**Output:**
- `config_suggested.yaml`: A new config file with suggested thresholds.
- `dist_*.png`: Plots showing the data distribution and calculated thresholds.

### 3. Single Scene Demo (`demo.py`)

Processes a single scene (the first one in the dataset) to quickly check if the system is working.

**Usage:**

```bash
uv run python demo.py
```

**Options:**
- No command-line arguments.
- Uses `config.yaml` for thresholds.
- Hardcoded to use `v1.0-mini` and default data path.

### 4. Benchmark Execution (`benchmark_execution.py`)

Measures the execution time of the classification logic across all available scenes to estimate performance.

**Usage:**

```bash
uv run python benchmark_execution.py
```

**Options:**
- No command-line arguments.
- Scans `../../data/nuscenes` for CAN bus data.

### 5. Verify Scenarios (`verify_new_scenarios.py`)

Runs a set of predefined test cases (unit tests) to verify the classifier logic against expected outcomes.

**Usage:**

```bash
uv run python verify_new_scenarios.py
```

**Options:**
- No command-line arguments.
- Uses `config.yaml` to run the tests.
