import sys
import os
import traceback
import datetime
import json
import cv2
import numpy as np

# Add current directory to sys.path to allow importing local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    from classifier import RuleBasedClassifier
except Exception as e:
    with open(os.path.join(current_dir, 'error_log.txt'), 'w') as f:
        f.write(f"Import Error: {traceback.format_exc()}")
    print("Import Error occurred. See error_log.txt")
    sys.exit(1)

def main():
    # Initialize NuScenes
    dataroot = 'c:\\Users\\chiba\\project\\DriveDataFilterExperiments\\data\\nuscenes'
    try:
        nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=True)
        nusc_can = NuScenesCanBus(dataroot=dataroot)
    except Exception as e:
        print(f"Error initializing NuScenes: {e}")
        return

    # Initialize Classifier
    config_path = os.path.join(current_dir, 'config.json')
    classifier = RuleBasedClassifier(config_path)

    # Output directory setup (Run ID)
    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    base_output_dir = os.path.join(current_dir, '..', 'output', run_id)
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"Base output directory created: {base_output_dir}")

    # Process first 10 scenes
    scenes_to_process = nusc.scene[:10]
    print(f"Processing {len(scenes_to_process)} scenes...")

    for i, scene in enumerate(scenes_to_process):
        scene_name = scene['name']
        scene_token = scene['token']
        print(f"[{i+1}/{len(scenes_to_process)}] Processing scene: {scene_name}")

        # Scene specific output directory
        scene_output_dir = os.path.join(base_output_dir, f"{scene_name}")
        os.makedirs(scene_output_dir, exist_ok=True)

        # Video writer setup
        output_video_path = os.path.join(scene_output_dir, 'demo_output.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None

        first_sample_token = scene['first_sample_token']
        current_sample_token = first_sample_token
        
        classification_data = {
            "scene_token": scene_token,
            "scene_name": scene_name,
            "samples": []
        }

        frame_count = 0
        
        while current_sample_token != '':
            sample = nusc.get('sample', current_sample_token)
            timestamp = sample['timestamp'] # Microseconds

            vehicle_state = get_vehicle_state(nusc_can, scene_name, timestamp)
            
            scenario = classifier._classify_frame(vehicle_state)
            
            # Collect structured data
            classification_data["samples"].append({
                "sample_token": current_sample_token,
                "timestamp": timestamp,
                "scenario": scenario,
                "vehicle_state": vehicle_state
            })
            
            # Visualization
            # Get camera image (CAM_FRONT)
            cam_front_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
            im_path = os.path.join(dataroot, cam_front_data['filename'])
            if not os.path.exists(im_path):
                print(f"Image not found: {im_path}")
                current_sample_token = sample['next']
                continue

            img = cv2.imread(im_path)
            
            if out is None:
                height, width, layers = img.shape
                out = cv2.VideoWriter(output_video_path, fourcc, 2, (width, height)) # 2 FPS approx

            # Overlay Text
            text = f"Scenario: {scenario}"
            cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Add vehicle state info for debug
            info_text = f"Speed: {vehicle_state.get('speed', 0):.2f} m/s, Steer: {vehicle_state.get('steering_angle', 0):.2f} rad"
            cv2.putText(img, info_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            out.write(img)
            frame_count += 1
            
            current_sample_token = sample['next']

        if out:
            out.release()
        print(f"  Video saved to {output_video_path}")
        
        # Save JSON output
        output_json_path = os.path.join(scene_output_dir, 'classification_results.json')
        with open(output_json_path, 'w') as f:
            json.dump(classification_data, f, indent=2)
        print(f"  Classification results saved to {output_json_path}")

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
    import bisect
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

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        with open(os.path.join(current_dir, 'error_log.txt'), 'w') as f:
            f.write(traceback.format_exc())
        print("Error occurred. See error_log.txt")
        sys.exit(1)
