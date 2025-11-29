import argparse
import os
import sys
import json
import time
import subprocess
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes

import datetime
import numpy as np
import bisect
from nuscenes.can_bus.can_bus_api import NuScenesCanBus

# Add current directory to sys.path to allow importing local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

def get_vehicle_state(nusc_can, scene_name, timestamp, tolerance=50000):
    """
    Find closest CAN messages to the timestamp.
    timestamp: microseconds
    tolerance: microseconds (default 50ms)
    """
    pose_msgs = nusc_can.get_messages(scene_name, 'pose')
    steer_msgs = nusc_can.get_messages(scene_name, 'steeranglefeedback')
    
    # Find message closest to timestamp
    pose = find_closest_msg(pose_msgs, timestamp, tolerance)
    steer = find_closest_msg(steer_msgs, timestamp, tolerance)
    
    state = {}
    if pose:
        state['speed'] = np.linalg.norm([pose['vel'][0], pose['vel'][1]]) # vel is [vx, vy, vz]
        state['yaw_rate'] = pose['rotation_rate'][2] # rotation_rate is [rx, ry, rz]
    
    if steer:
        state['steering_angle'] = steer['value']

    # Turn signal is in 'vehicle_monitor' usually, but might not be populated in all datasets.
    # We'll default to 0.
    state['turn_signal'] = 0 
    
    return state

def find_closest_msg(msgs, timestamp, tolerance):
    if not msgs:
        return None
    
    # Binary search for closest
    times = [m['utime'] for m in msgs]
    idx = bisect.bisect_left(times, timestamp)
    
    candidates = []
    if idx < len(msgs):
        candidates.append(msgs[idx])
    if idx > 0:
        candidates.append(msgs[idx-1])
        
    best_msg = None
    min_diff = float('inf')
    
    for msg in candidates:
        diff = abs(msg['utime'] - timestamp)
        if diff < min_diff and diff <= tolerance:
            min_diff = diff
            best_msg = msg
            
    return best_msg

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Ground Truth Ego Behavior Labels using Gemini CLI.")
    parser.add_argument("--version", type=str, default="v1.0-mini", help="NuScenes version (e.g., v1.0-mini, v1.0-trainval)")
    parser.add_argument("--dataroot", type=str, required=True, help="Path to NuScenes data root")
    parser.add_argument("--output", type=str, default="gemini_labels.json", help="Output JSON file path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to process")
    parser.add_argument("--scene_name", type=str, default=None, help="Specific scene name to process")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="Gemini model to use (e.g., gemini-2.5-flash, gemini-2.5-flash-lite)")
    parser.add_argument("--prompt", type=str, default="Describe this image and classify it according to the definitions. Output JSON.", help="Prompt to send to Gemini CLI")
    return parser.parse_args()

def run_gemini_cli(prompt, images, model):
    """
    Runs the gemini CLI command.
    """
    cmd = ["gemini", "-m", model, prompt] + images
    print(f"Running command: {cmd}")
    
    # Run command
    try:
        # We need to run this in the directory where GEMINI.md resides to pick up context.
        # Assuming GEMINI.md is in the project root.
        cwd = os.getcwd() 
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            encoding='utf-8',
            cwd=cwd,
            shell=True,
            timeout=30 # 30 seconds timeout
        )
        print(f"Command finished with return code: {result.returncode}")
        
        if result.returncode != 0:
            print(f"Error running Gemini CLI: {result.stderr}")
            return None
            
        return result.stdout
    except Exception as e:
        print(f"Exception running Gemini CLI: {e}")
        return None

def extract_json(text):
    """
    Extracts JSON from the text response.
    """
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].strip()
        
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find { ... }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end+1])
            except:
                pass
        return None

def main():
    args = parse_args()

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
            
            images = [cam_front_path]

            # Call Gemini CLI
            response_text = run_gemini_cli(args.prompt, images, args.model)
            
            if response_text:
                data = extract_json(response_text)
                if data:
                    result_entry = {
                        "sample_token": current_sample_token,
                        "timestamp": timestamp,
                        "scene_name": scene_name,
                        "gemini_label": data
                    }
                    results.append(result_entry)
                    print(f"[{processed_count+1}] {scene_name} - {data.get('class_name', 'Unknown')}")
                else:
                    print(f"[{processed_count+1}] Failed to parse JSON: {response_text[:100]}...")
            else:
                print(f"[{processed_count+1}] No response from Gemini CLI.")

            processed_count += 1
            current_sample_token = sample['next']
            
            # Rate limiting? CLI might handle it or we might need to sleep.
            # time.sleep(1) 
        
        if args.limit and processed_count >= args.limit:
            break

def main():
    args = parse_args()

    # Initialize NuScenes
    print(f"Initializing NuScenes {args.version}...")
    try:
        nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
        nusc_can = NuScenesCanBus(dataroot=args.dataroot)
    except Exception as e:
        print(f"Error initializing NuScenes: {e}")
        return

    # Output directory setup
    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(current_dir, '..', 'output', run_id)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")
    
    output_path = os.path.join(output_dir, 'gemini_labels.json')

    final_results = []
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
        scene_token = scene['token']
        first_sample_token = scene['first_sample_token']
        current_sample_token = first_sample_token

        scene_data = {
            "scene_token": scene_token,
            "scene_name": scene_name,
            "samples": []
        }

        while current_sample_token != '':
            if args.limit and processed_count >= args.limit:
                break

            sample = nusc.get('sample', current_sample_token)
            timestamp = sample['timestamp']

            # Get CAM_FRONT image
            cam_front_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
            cam_front_path = os.path.join(args.dataroot, cam_front_data['filename'])
            
            # Get CAM_BACK image (optional)
            images = [cam_front_path]
            if 'CAM_BACK' in sample['data']:
                cam_back_data = nusc.get('sample_data', sample['data']['CAM_BACK'])
                cam_back_path = os.path.join(args.dataroot, cam_back_data['filename'])
                images.append(cam_back_path)

            # Get Vehicle State
            vehicle_state = get_vehicle_state(nusc_can, scene_name, timestamp)

            # Call Gemini CLI
            response_text = run_gemini_cli(args.prompt, images, args.model)
            
            if response_text:
                data = extract_json(response_text)
                if data:
                    # Construct sample object according to data_format.md
                    sample_entry = {
                        "sample_token": current_sample_token,
                        "timestamp": timestamp,
                        "scenario": data.get('class_name', 'Unknown'),
                        "vehicle_state": vehicle_state,
                        "reasoning": data.get('reasoning', '') # Added reasoning as extra field, though not in strict spec, it's useful.
                    }
                    scene_data["samples"].append(sample_entry)
                    print(f"[{processed_count+1}] {scene_name} - {data.get('class_name', 'Unknown')}")
                else:
                    print(f"[{processed_count+1}] Failed to parse JSON: {response_text[:100]}...")
            else:
                print(f"[{processed_count+1}] No response from Gemini CLI.")

            processed_count += 1
            current_sample_token = sample['next']
            
        if scene_data["samples"]:
            final_results.append(scene_data)
        
        if args.limit and processed_count >= args.limit:
            break

    # Save results
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"Saved labels to {output_path}")

if __name__ == "__main__":
    main()
