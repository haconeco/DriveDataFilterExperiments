import argparse
import os
import sys
import json
import time
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes

# Add current directory to sys.path to allow importing local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from prompts import SYSTEM_PROMPT
from utils import setup_gemini, load_image

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Ground Truth Ego Behavior Labels using Gemini.")
    parser.add_argument("--version", type=str, default="v1.0-mini", help="NuScenes version (e.g., v1.0-mini, v1.0-trainval)")
    parser.add_argument("--dataroot", type=str, required=True, help="Path to NuScenes data root")
    parser.add_argument("--output", type=str, default="gemini_labels.json", help="Output JSON file path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to process")
    parser.add_argument("--model", type=str, default="gemini-1.5-flash", help="Gemini model name")
    parser.add_argument("--scene_name", type=str, default=None, help="Specific scene name to process")
    return parser.parse_args()

def main():
    args = parse_args()

    # Initialize Gemini
    try:
        model = setup_gemini(model_name=args.model)
        print(f"Gemini model '{args.model}' initialized.")
    except Exception as e:
        print(f"Error initializing Gemini: {e}")
        return

    # Initialize NuScenes
    print(f"Initializing NuScenes {args.version}...")
    try:
        nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    except Exception as e:
        print(f"Error initializing NuScenes: {e}")
        return

    results = []
    processed_count = 0
    
    # Filter scenes if needed
    scenes = nusc.scene
    if args.scene_name:
        scenes = [s for s in scenes if s['name'] == args.scene_name]
        if not scenes:
            print(f"Scene '{args.scene_name}' not found.")
            return

    print(f"Processing {len(scenes)} scenes...")

    for scene in scenes:
        scene_name = scene['name']
        first_sample_token = scene['first_sample_token']
        current_sample_token = first_sample_token

        while current_sample_token != '':
            if args.limit and processed_count >= args.limit:
                break

            sample = nusc.get('sample', current_sample_token)
            timestamp = sample['timestamp']

            # Get CAM_FRONT image
            cam_front_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
            cam_front_path = os.path.join(args.dataroot, cam_front_data['filename'])
            
            # Get CAM_BACK image (optional, but good for context)
            cam_back_data = nusc.get('sample_data', sample['data']['CAM_BACK'])
            cam_back_path = os.path.join(args.dataroot, cam_back_data['filename'])

            images = []
            try:
                images.append(load_image(cam_front_path))
                images.append(load_image(cam_back_path))
            except Exception as e:
                print(f"Error loading images for sample {current_sample_token}: {e}")
                current_sample_token = sample['next']
                continue

            # Construct prompt
            # We pass images and the system prompt.
            # Note: Gemini API python client handles image + text list.
            content = [SYSTEM_PROMPT, "Front Camera:", images[0], "Back Camera:", images[1]]

            try:
                response = model.generate_content(content)
                # Parse JSON response
                # Gemini might return markdown code block ```json ... ```
                text = response.text
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].strip()
                
                data = json.loads(text)
                
                result_entry = {
                    "sample_token": current_sample_token,
                    "timestamp": timestamp,
                    "scene_name": scene_name,
                    "gemini_label": data
                }
                results.append(result_entry)
                print(f"[{processed_count+1}] {scene_name} - {data.get('class_name', 'Unknown')}")

            except Exception as e:
                print(f"Error processing sample {current_sample_token}: {e}")
                # Optional: Sleep to avoid rate limits if hitting them hard
                time.sleep(1)

            processed_count += 1
            current_sample_token = sample['next']
            
            # Rate limiting / Sleep to be safe (adjust as needed)
            time.sleep(1) 
        
        if args.limit and processed_count >= args.limit:
            break

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} labels to {args.output}")

if __name__ == "__main__":
    main()
