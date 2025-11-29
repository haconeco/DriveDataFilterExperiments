# SegFormer Environment Setup Walkthrough

## 1. Environment Build
The Docker environment was successfully built.
- **Dockerfile**: Modified to fix dependencies (`numpy<2.0`, `ftfy`, `regex`).
- **docker-compose.yml**: Modified to mount `tests` and `tools` directories.

Command:
```bash
docker-compose build
```

## 2. Unit Tests
Unit tests were executed to verify the environment and inference logic.
- **Status**: Passed
- **Tests Run**: `tests/test_inference.py` (and others if present)

Command:
```bash
docker-compose run --rm segformer python3 -m unittest discover tests
```

## 3. Inference Execution
Inference was successfully executed on NuScenes data using the provided weights and local configuration.
- **Status**: Success
- **Weights**: `weights/iter_160000.pth`
- **Config**: `../RoadLib/scripts/segformer_whu.py` (mounted)
- **Command**:
```bash
docker-compose run --rm segformer python3 tools/inference.py \
  --config mmsegmentation/configs/segformer/segformer_whu.py \
  --checkpoint weights/iter_160000.pth \
  --dataroot data/nuscenes \
  --version v1.0-mini \
  --output_dir output/run_02 \
  --visualize
```

### Results
- **JSON Output**: `output/run_02/results.json` (Contains detected instances with bounding boxes and polygons)
- **Visualizations**: `output/run_02/vis/` (Overlay images)
- **Masks**: `output/run_02/masks/` (Segmentation masks)

## 4. Performance Verification
Inference was executed without visualization to measure processing speed.
- **Command**:
```bash
docker-compose run --rm segformer python3 tools/inference.py \
  --config mmsegmentation/configs/segformer/segformer_whu.py \
  --checkpoint weights/iter_160000.pth \
  --dataroot data/nuscenes \
  --version v1.0-mini \
  --output_dir output/run_03_no_vis
```
- **Results**:
  - **Time per Sample (6 images)**: ~2.02 seconds
  - **Time per Image**: ~0.34 seconds
  - **FPS (per image)**: ~2.9 FPS
- **Output**: `output/run_03_no_vis/results.json` (Generated successfully)

## 5. Performance Verification (No Mask)
Inference was executed without visualization AND without saving masks to measure pure inference speed.
- **Command**:
```bash
docker-compose run --rm segformer python3 tools/inference.py \
  --config mmsegmentation/configs/segformer/segformer_whu.py \
  --checkpoint weights/iter_160000.pth \
  --dataroot data/nuscenes \
  --version v1.0-mini \
  --output_dir output/run_04_no_vis_no_mask \
  --no_save_mask
```
- **Results**:
  - **Time per Sample (6 images)**: ~1.98 seconds
  - **Time per Image**: ~0.33 seconds
  - **FPS (per image)**: ~3.0 FPS
- **Output**: `output/run_04_no_vis_no_mask/results.json` (Generated successfully)
- **Masks**: `output/run_04_no_vis_no_mask/masks/` (Empty, as expected)

## 6. Performance Verification (Batch Size 3)
Inference was executed with `--batch_size 3` and `--no_save_mask` to verify performance improvement with parallel processing.
- **Command**:
```bash
docker-compose run --rm segformer python3 tools/inference.py \
  --config mmsegmentation/configs/segformer/segformer_whu.py \
  --checkpoint weights/iter_160000.pth \
  --dataroot data/nuscenes \
  --version v1.0-mini \
  --output_dir output/run_05_batch3 \
  --no_save_mask \
  --batch_size 3
```
- **Results**:
  - **Time per Sample (6 images)**: ~1.86 seconds
  - **Time per Image**: ~0.31 seconds
  - **FPS (per image)**: ~3.2 FPS
- **Output**: `output/run_05_batch3/results.json` (Generated successfully)
