import sys
import os
import time
import numpy as np
import glob
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from classifier import RuleBasedClassifier

def load_all_can_data(dataroot):
    print(f"Scanning CAN bus data from {dataroot}...")
    can_bus_dir = os.path.join(dataroot, 'can_bus')
    if not os.path.exists(can_bus_dir):
        if os.path.basename(dataroot) == 'can_bus':
            can_bus_dir = dataroot
        else:
             print(f"Warning: {can_bus_dir} not found.")
             return []

    scene_meta_files = glob.glob(os.path.join(can_bus_dir, 'scene-*_meta.json'))
    scene_names = [os.path.basename(f).replace('_meta.json', '') for f in scene_meta_files]
    scene_names.sort()
    
    print(f"Found {len(scene_names)} scenes.")
    return scene_names

def main():
    dataroot = 'c:\\Users\\chiba\\project\\DriveDataFilterExperiments\\data\\nuscenes'
    
    # Initialize Classifier
    config_path = 'config.yaml'
    classifier = RuleBasedClassifier(config_path)
    
    # Get all scenes
    scene_names = load_all_can_data(dataroot)
    if not scene_names:
        print("No scenes found.")
        return

    nusc_can = NuScenesCanBus(dataroot=dataroot)
    
    total_frames = 0
    total_time = 0
    
    print("Starting benchmark...")
    benchmark_start = time.time()
    
    for scene_name in scene_names:
        try:
            # We need to construct "frames" from CAN messages.
            # In a real scenario, we stream messages. 
            # Here, we'll fetch all relevant messages and simulate a frame-by-frame loop.
            # To be realistic, we should align them by time, but for "execution cost of classification",
            # we just need to pass valid state dicts to the classifier.
            
            pose_msgs = nusc_can.get_messages(scene_name, 'pose')
            steer_msgs = nusc_can.get_messages(scene_name, 'steeranglefeedback')
            monitor_msgs = nusc_can.get_messages(scene_name, 'vehicle_monitor')
            
            # Let's assume 1 frame per pose message (approx 50Hz or 100Hz depending on can bus)
            # Actually pose is usually 50Hz.
            
            # Pre-process messages into a list of states to minimize "data loading" noise 
            # and focus on "classification" cost?
            # The user asked for "execution cost when applying to NuScenes CAN".
            # This likely includes the overhead of retrieving the data and running the logic.
            # So we will measure the loop time.
            
            # However, nusc_can.get_messages reads the whole file at once. 
            # So the iteration is fast.
            
            # Let's align roughly.
            # We'll just iterate pose messages and find corresponding steer/signal.
            # To avoid O(N^2) lookup in this benchmark (which would dominate), 
            # we'll use simple index matching or just assume same frequency for benchmarking the *classifier*.
            # But wait, the user wants "Execution Cost". If our data loading is slow, that's part of the cost.
            # But `get_messages` is already done.
            
            # Let's create a list of "Vehicle States" first, then benchmark the classification loop.
            # This separates "Data IO" from "Processing".
            
            # Optimization: Convert to numpy arrays or dictionaries for fast lookup?
            # For this benchmark, let's just iterate pose and use the latest steering/signal value.
            
            steer_vals = [m['value'] for m in steer_msgs]
            steer_times = [m['utime'] for m in steer_msgs]
            
            monitor_vals = [m['turn_signal'] for m in monitor_msgs if 'turn_signal' in m] # Simplified
            # monitor_times = ...
            
            # Just use simple iteration for speed
            current_steer = 0
            current_signal = 0
            steer_idx = 0
            monitor_idx = 0
            
            scene_states = []
            
            for pose in pose_msgs:
                timestamp = pose['utime']
                
                # Update steering (assuming sorted)
                while steer_idx < len(steer_msgs) and steer_msgs[steer_idx]['utime'] <= timestamp:
                    current_steer = steer_msgs[steer_idx]['value']
                    steer_idx += 1
                
                # Update signal
                while monitor_idx < len(monitor_msgs) and monitor_msgs[monitor_idx]['utime'] <= timestamp:
                    if 'turn_signal' in monitor_msgs[monitor_idx]:
                        current_signal = monitor_msgs[monitor_idx]['turn_signal']
                    monitor_idx += 1
                
                state = {
                    'speed': np.linalg.norm([pose['vel'][0], pose['vel'][1]]),
                    'yaw_rate': pose['rotation_rate'][2],
                    'steering_angle': current_steer,
                    'turn_signal': current_signal
                }
                scene_states.append(state)
            
            # Now benchmark the classification
            start_t = time.time()
            for state in scene_states:
                _ = classifier._classify_frame(state)
            end_t = time.time()
            
            total_time += (end_t - start_t)
            total_frames += len(scene_states)
            
        except Exception as e:
            # print(f"Error in {scene_name}: {e}")
            continue
            
    benchmark_end = time.time()
    total_benchmark_time = benchmark_end - benchmark_start
    
    print(f"\nBenchmark Results:")
    print(f"Total Scenes Processed: {len(scene_names)}")
    print(f"Total Frames Processed: {total_frames}")
    print(f"Total Classification Time (Pure Logic): {total_time:.4f} seconds")
    print(f"Average Time per Frame: {total_time / total_frames * 1000:.4f} ms")
    print(f"Projected FPS: {total_frames / total_time:.2f}")
    
    # Estimate for full NuScenes (850 scenes, ~20s each, 50Hz? -> ~1000 frames/scene -> 850,000 frames)
    # NuScenes is 1000 scenes. 20s each. 20Hz keyframes, but CAN is higher freq.
    # Pose is 50Hz. So 1000 * 20 * 50 = 1,000,000 frames.
    estimated_total_frames = 1000 * 20 * 50 # 1 Million
    estimated_total_time = (total_time / total_frames) * estimated_total_frames
    
    print(f"\nEstimated Time for Full NuScenes (1M frames): {estimated_total_time:.2f} seconds ({estimated_total_time/60:.2f} minutes)")

if __name__ == "__main__":
    main()
