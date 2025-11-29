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

    def process_image(self, img_path):
        """
        Run inference on a single image.
        Returns:
            result (dict): Inference result containing 'predictions' (mask).
            vis_image (np.ndarray): Visualization image (if requested by inferencer, but we do manual vis).
        """
        # MMSegInferencer returns a dictionary with 'predictions' key which holds the mask
        # The return format depends on return_datasamples. 
        # By default (return_datasamples=False), it returns a dict with 'predictions' (mask) and 'visualization'
        result = self.inferencer(img_path, return_datasamples=False)
        return result

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
    for sample in tqdm(nusc.sample):
        sample_token = sample['token']
        results[sample_token] = {}
        
        for cam_name in cameras:
            if cam_name not in sample['data']:
                continue
                
            cam_token = sample['data'][cam_name]
            cam_data = nusc.get('sample_data', cam_token)
            
            # Get absolute path
            img_path = Path(nusc.get_sample_data_path(cam_token))
            filename = img_path.name
            
            # Inference
            inference_result = detector.process_image(str(img_path))
            
            # Extract mask (MMSegInferencer returns dict with 'predictions' key)
            # predictions is usually a numpy array of shape (H, W)
            mask = inference_result['predictions']
            
            # Extract instances
            instances = detector.extract_instances(mask)
            
            # Save mask
            mask_filename = f"{filename.replace('.jpg', '.png')}"
            mask_path = mask_dir / mask_filename
            cv2.imwrite(str(mask_path), mask)
            
            # Visualization
            if args.visualize:
                vis_path = vis_dir / f"vis_{filename}"
                detector.visualize(img_path, mask, vis_path)
            
            # Store results
            results[sample_token][cam_name] = {
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
