# Rule-Based Scenario Labeling Demo Walkthrough

## Overview
This walkthrough describes the process and results of generating rule-based scenario labeling for 10 NuScenes scenes.

## Changes Made
- Created `canbus_scenalializer/rule_based/generate_demo_scenes.py` to process 10 scenes.
- The script uses `RuleBasedClassifier` to classify ego vehicle behavior based on CAN bus data.
- Generates a video overlay and a JSON file for each scene.

## Verification Results
- **Script Execution**: Successfully ran `generate_demo_scenes.py`.
- **Output Directory**: `canbus_scenalializer/output/20251201_234915/`
- **Generated Files**:
    - 10 scene directories (e.g., `scene-0061`, `scene-0103`).
    - Each directory contains:
        - `demo_output.mp4`: Visualization video with scenario overlay.
        - `classification_results.json`: JSON file with classification results.
- **JSON Validation**: Verified that `classification_results.json` adheres to the defined schema.

## Example Output
### JSON Snippet (scene-0061)
```json
{
  "scene_token": "cc8c0bf57f984915a77078b10eb33198",
  "scene_name": "scene-0061",
  "samples": [
    {
      "sample_token": "ca9a282c9e77460f8360f564131a8af5",
      "timestamp": 1532402927647951,
      "scenario": "Cruising",
      "vehicle_state": {
        "speed": 8.988725931436823,
        "yaw_rate": 0.017157725989818573,
        "steering_angle": 0.04712388980384372,
        "turn_signal": 0
      }
    },
    ...
  ]
}
```
