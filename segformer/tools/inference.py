import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from mmseg.apis import MMSegInferencer
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm

# RoadLib Class Definitions
# 0: Background (EMPTY)
# 1: SOLID
# 2: DASHED
# 3: GUIDE
# 4: ZEBRA
# 5: STOP
CLASSES = {
    0: "background",
    1: "solid",
    2: "dashed",
    3: "guide",
    4: "zebra",
    5: "stop"
}

class RoadMarkingDetector:
    def __init__(self, config_path, checkpoint_path, device='cuda'):
        self.inferencer = MMSegInferencer(
            model=config_path,
            weights=checkpoint_path,
            device=device
        )

    def process_batch(self, img_paths, batch_size=1):
        """
        Run inference on a batch of images.
        Args:
            img_paths (list): List of image paths.
            batch_size (int): Batch size.
        Returns:
            results (list): List of inference results.
        """
        # MMSegInferencer can handle list of inputs
        # It returns a dictionary with 'predictions' which is a list of masks if input is a list
        results = self.inferencer(img_paths, batch_size=batch_size, return_datasamples=False)
        
        # If input is a list, output 'predictions' is a list of masks
        # We need to structure it as a list of dicts to match single image output format for consistency
        # However, MMSegInferencer return format might vary slightly.
        # Based on source:
        # results_dict['predictions'] = []
        # ...
        # results_dict['predictions'].append(pred_data)
        # return results_dict
        
        # So results['predictions'] should be a list of masks.
        
        formatted_results = []
        predictions = results['predictions']
        
        # Handle case where single image input returns single item not list
        if not isinstance(predictions, list):
            predictions = [predictions]
            
        for pred in predictions:
            formatted_results.append({'predictions': pred})
            
        return formatted_results

    def extract_instances(self, mask):
        """
        Extract instances from segmentation mask using connected components / contours.
        Args:
            mask (np.ndarray): Segmentation mask (H, W) with class IDs.
        Returns:
            instances (list): List of dicts containing label, score (dummy), bbox, polygon.
        """
        instances = []
        
        # Iterate over each class (skipping background 0)
        for class_id in range(1, 6):
            if class_id not in CLASSES:
                continue
                
            # Create binary mask for the current class
            class_mask = (mask == class_id).astype(np.uint8)
            
            if np.sum(class_mask) == 0:
                continue

            # Find contours
            contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) < 50: # Filter small noise
                    continue
                    
                # Bounding box
                x, y, w, h = cv2.boundingRect(contour)
                bbox = [int(x), int(y), int(w), int(h)]
                
                # Polygon (flattened list of points)
                polygon = contour.flatten().tolist()
                
                instances.append({
                    "label": CLASSES[class_id],
                    "class_id": class_id,
                    "score": 1.0, # Placeholder as semantic seg doesn't give instance scores
                    "bbox": bbox,
                    "polygon": polygon
                })
                
        return instances

    def visualize(self, img_path, mask, output_path):
        """
        Save overlay image.
        """
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            return

        # Create colored mask
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Define colors for classes (BGR)
        colors = {
            1: (0, 0, 255),    # Solid: Red
            2: (0, 255, 255),  # Dashed: Yellow
            3: (255, 0, 0),    # Guide: Blue
            4: (255, 0, 255),  # Zebra: Magenta
            5: (0, 255, 0)     # Stop: Green
        }
        
        for class_id, color in colors.items():
            color_mask[mask == class_id] = color
            
        # Blend
        alpha = 0.5
        overlay = cv2.addWeighted(img, 1 - alpha, color_mask, alpha, 0)
        
        cv2.imwrite(str(output_path), overlay)

def main():
    parser = argparse.ArgumentParser(description="SegFormer Inference on NuScenes")
    parser.add_argument("--config", required=True, help="Path to model config")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--dataroot", required=True, help="NuScenes dataroot")
    parser.add_argument("--version", default="v1.0-mini", help="NuScenes version")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    parser.add_argument("--no_save_mask", action="store_true", help="Disable mask saving")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    args = parser.parse_args()

    # Initialize NuScenes
    print(f"Initializing NuScenes ({args.version})...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    # Initialize Detector
    print("Loading model...")
    detector = RoadMarkingDetector(args.config, args.checkpoint)

    # Prepare output directories
    output_dir = Path(args.output_dir)
    mask_dir = output_dir / "masks"
    vis_dir = output_dir / "vis"
    mask_dir.mkdir(parents=True, exist_ok=True)
    if args.visualize:
        vis_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    
    cameras = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

    print("Starting inference...")
    print("Starting inference...")
    
    # Collect all tasks first to batch them
    tasks = []
    for sample in nusc.sample:
        sample_token = sample['token']
        results[sample_token] = {}
        
        for cam_name in cameras:
            if cam_name not in sample['data']:
                continue
            
            cam_token = sample['data'][cam_name]
            img_path = Path(nusc.get_sample_data_path(cam_token))
            
            tasks.append({
                'sample_token': sample_token,
                'cam_name': cam_name,
                'img_path': str(img_path),
                'filename': img_path.name
            })

    # Process in batches
    for i in tqdm(range(0, len(tasks), args.batch_size)):
        batch_tasks = tasks[i:i + args.batch_size]
        batch_paths = [t['img_path'] for t in batch_tasks]
        
        # Inference
        batch_results = detector.process_batch(batch_paths, batch_size=args.batch_size)
        
        for j, task in enumerate(batch_tasks):
            inference_result = batch_results[j]
            mask = inference_result['predictions']
            
            # Extract instances
            instances = detector.extract_instances(mask)
            
            filename = task['filename']
            mask_filename = f"{filename.replace('.jpg', '.png')}"
            
            # Save mask
            if not args.no_save_mask:
                mask_path = mask_dir / mask_filename
                cv2.imwrite(str(mask_path), mask)
            
            # Visualization
            if args.visualize:
                vis_path = vis_dir / f"vis_{filename}"
                detector.visualize(task['img_path'], mask, vis_path)
            
            # Store results
            results[task['sample_token']][task['cam_name']] = {
                "filename": filename,
                "mask_path": str(mask_filename),
                "instances": instances
            }

    # Save results JSON
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Done. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
