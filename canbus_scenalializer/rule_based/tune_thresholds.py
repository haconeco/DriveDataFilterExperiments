import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus

# Add current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import glob

def load_can_only_data(dataroot):
    print(f"Scanning CAN bus data directly from {dataroot}...")
    can_bus_dir = os.path.join(dataroot, 'can_bus')
    if not os.path.exists(can_bus_dir):
        # Try checking if dataroot itself is the can_bus dir or contains it differently
        if os.path.basename(dataroot) == 'can_bus':
            can_bus_dir = dataroot
        else:
             print(f"Warning: {can_bus_dir} not found.")
             return np.array([]), np.array([]), np.array([])

    # Find all scene meta files
    scene_meta_files = glob.glob(os.path.join(can_bus_dir, 'scene-*_meta.json'))
    scene_names = [os.path.basename(f).replace('_meta.json', '') for f in scene_meta_files]
    scene_names.sort()
    
    print(f"Found {len(scene_names)} scenes in {can_bus_dir}")
    
    nusc_can = NuScenesCanBus(dataroot=dataroot)
    
    speeds = []
    steerings = []
    yaw_rates = []
    
    for scene_name in scene_names:
        try:
            # Get all pose messages for speed and yaw rate
            pose_msgs = nusc_can.get_messages(scene_name, 'pose')
            for msg in pose_msgs:
                vel = msg['vel'] # [vx, vy, vz]
                speed = np.linalg.norm([vel[0], vel[1]])
                speeds.append(speed)
                
                rot_rate = msg['rotation_rate'] # [rx, ry, rz]
                yaw_rate = abs(rot_rate[2])
                yaw_rates.append(yaw_rate)
                
            # Get all steer messages
            steer_msgs = nusc_can.get_messages(scene_name, 'steeranglefeedback')
            for msg in steer_msgs:
                steering = abs(msg['value'])
                steerings.append(steering)
        except Exception as e:
            print(f"Error processing {scene_name}: {e}")
            continue
            
    return np.array(speeds), np.array(steerings), np.array(yaw_rates)

def load_data(dataroot, version='v1.0-mini'):
    print(f"Loading NuScenes from {dataroot}...")
    try:
        nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        scenes = [s['name'] for s in nusc.scene]
    except Exception as e:
        print(f"NuScenes initialization failed: {e}")
        print("Falling back to CAN-only mode...")
        return load_can_only_data(dataroot)

    nusc_can = NuScenesCanBus(dataroot=dataroot)
    
    speeds = []
    steerings = []
    yaw_rates = []
    
    print("Extracting CAN bus data...")
    for scene_name in scenes:
        try:
            # Get all pose messages for speed and yaw rate
            pose_msgs = nusc_can.get_messages(scene_name, 'pose')
            for msg in pose_msgs:
                vel = msg['vel'] # [vx, vy, vz]
                speed = np.linalg.norm([vel[0], vel[1]])
                speeds.append(speed)
                
                rot_rate = msg['rotation_rate'] # [rx, ry, rz]
                yaw_rate = abs(rot_rate[2])
                yaw_rates.append(yaw_rate)
                
            # Get all steer messages
            steer_msgs = nusc_can.get_messages(scene_name, 'steeranglefeedback')
            for msg in steer_msgs:
                steering = abs(msg['value'])
                steerings.append(steering)
        except Exception as e:
            # Some scenes might not have CAN bus data
            # print(f"Skipping {scene_name}: {e}")
            continue
            
    return np.array(speeds), np.array(steerings), np.array(yaw_rates)

def fit_gmm_and_find_thresholds(data, n_components, sigma=2.0):
    # Reshape for sklearn
    X = data.reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X)
    
    # Sort components by mean
    means = gmm.means_.flatten()
    sorted_indices = np.argsort(means)
    
    thresholds = []
    
    # For each component i (except the last one), find the boundary to the next component
    # We define the boundary as mean_i + sigma * std_i
    for i in range(n_components - 1):
        idx = sorted_indices[i]
        mean = means[idx]
        cov = gmm.covariances_.flatten()[idx]
        std = np.sqrt(cov)
        
        threshold = mean + sigma * std
        thresholds.append(threshold)
    
    return thresholds, gmm

def plot_distribution(data, gmm, thresholds, title, xlabel, filename):
    plt.figure(figsize=(10, 6))
    
    # Plot histogram
    plt.hist(data, bins=100, density=True, alpha=0.6, color='g', label='Data Histogram')
    
    # Plot GMM PDF
    x = np.linspace(0, np.max(data), 1000)
    logprob = gmm.score_samples(x.reshape(-1, 1))
    pdf = np.exp(logprob)
    plt.plot(x, pdf, '-k', label='GMM PDF')
    
    # Plot individual components
    for i in range(gmm.n_components):
        mean = gmm.means_.flatten()[i]
        var = gmm.covariances_.flatten()[i]
        weight = gmm.weights_.flatten()[i]
        pdf_comp = weight * (1 / np.sqrt(2 * np.pi * var)) * np.exp(- (x - mean)**2 / (2 * var))
        plt.plot(x, pdf_comp, '--', label=f'Component {i+1}')

    # Plot thresholds
    colors = ['r', 'b', 'm', 'c']
    for i, threshold in enumerate(thresholds):
        color = colors[i % len(colors)]
        plt.axvline(threshold, color=color, linestyle='dashed', linewidth=2, label=f'Threshold {i+1}: {threshold:.3f}')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.close()

import argparse

def main():
    parser = argparse.ArgumentParser(description="Tune CAN bus thresholds using GMM.")
    parser.add_argument('--dataroot', type=str, default='c:\\Users\\chiba\\project\\DriveDataFilterExperiments\\data\\nuscenes', help='Path to NuScenes data root')
    parser.add_argument('--version', type=str, default='v1.0-mini', help='NuScenes version (e.g., v1.0-mini, v1.0-trainval)')
    args = parser.parse_args()

    dataroot = args.dataroot
    version = args.version
    
    print(f"Running with dataroot: {dataroot}, version: {version}")

    speeds, steerings, yaw_rates = load_data(dataroot, version=version)
    
    print(f"Data loaded: {len(speeds)} speed samples, {len(steerings)} steering samples, {len(yaw_rates)} yaw samples.")
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Speed Thresholds (Stop -> Slow/Pull Over -> Cruising)
    # Fit 3 components
    print("Fitting Speed GMM (3 components)...")
    speed_thresholds, speed_gmm = fit_gmm_and_find_thresholds(speeds, n_components=3)
    stop_speed_threshold = speed_thresholds[0]
    pull_over_speed_threshold = speed_thresholds[1]
    
    plot_distribution(speeds, speed_gmm, speed_thresholds, 'Speed Distribution', 'Speed (m/s)', os.path.join(output_dir, 'dist_speed.png'))
    
    # 2. Steering Thresholds (Straight -> Lane Change -> Turn -> U-Turn)
    # Fit 3 components: Straight, Turn, U-Turn
    # We want to capture "Normal Turns" which might be the lower end of the "Turn" component.
    print("Fitting Steering GMM (3 components)...")
    steer_thresholds, steer_gmm = fit_gmm_and_find_thresholds(steerings, n_components=3)
    
    # Threshold 0: Straight -> Turn (Use this for Lane Change start?)
    # Threshold 1: Turn -> U-Turn
    
    # Let's define:
    # Lane Change Threshold = Threshold 0 (Upper bound of Straight)
    # Turn Threshold = Threshold 0 (Any non-straight is a turn candidate?)
    # But we want to distinguish Lane Change from Turn.
    # Lane Change is usually small steering. Turn is larger.
    # If we only have 3 components, Component 1 is "Turn".
    # Maybe we can set Turn Threshold to be slightly inside Component 1?
    # Or just use Threshold 0 for Lane Change, and (Threshold 0 + Mean 1)/2 for Turn?
    
    lane_change_steering_threshold = steer_thresholds[0]
    
    means = steer_gmm.means_.flatten()
    sorted_indices = np.argsort(means)
    turn_idx = sorted_indices[1]
    turn_mean = means[turn_idx]
    
    # Heuristic: Turn starts halfway between Straight limit and Turn mean?
    # Or just use Threshold 0 + small buffer?
    # Let's try: Turn Threshold = Lane Change Threshold * 1.5?
    # Or let's look at the gap.
    # If Threshold 0 is 0.3 and Mean 1 is 1.5.
    # Lane Change is > 0.3. Turn is > ?
    # Let's set Turn Threshold to (Threshold 0 + Turn Mean) / 2.
    turn_steering_threshold = (lane_change_steering_threshold + turn_mean) / 2
    
    u_turn_steering_threshold = steer_thresholds[1]
    
    plot_distribution(steerings, steer_gmm, [lane_change_steering_threshold, turn_steering_threshold, u_turn_steering_threshold], 'Steering Angle Distribution', 'Steering Angle (rad)', os.path.join(output_dir, 'dist_steering.png'))
    
    # 3. Yaw Rate Threshold (Straight -> Turn)
    # Fit 2 components (Keep simple for now, or 3 if we want sharp turns)
    print("Fitting Yaw Rate GMM (2 components)...")
    yaw_thresholds, yaw_gmm = fit_gmm_and_find_thresholds(yaw_rates, n_components=2)
    yaw_rate_threshold = yaw_thresholds[0]
    
    plot_distribution(yaw_rates, yaw_gmm, yaw_thresholds, 'Yaw Rate Distribution', 'Yaw Rate (rad/s)', os.path.join(output_dir, 'dist_yaw.png'))
    
    # Generate Config
    new_config = {
        "thresholds": {
            "stop_speed_threshold": float(stop_speed_threshold),
            "pull_over_speed_threshold": float(pull_over_speed_threshold),
            "lane_change_steering_threshold": float(lane_change_steering_threshold),
            "turn_steering_threshold": float(turn_steering_threshold),
            "u_turn_steering_threshold": float(u_turn_steering_threshold),
            "yaw_rate_threshold": float(yaw_rate_threshold),
            "turn_signal_on_threshold": 0.5 # Keep default
        }
    }
    
    config_out_path = os.path.join(output_dir, 'config_suggested.yaml')
    import yaml
    with open(config_out_path, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False)
        
    print(f"Suggested config saved to {config_out_path}")
    print("Thresholds:")
    print(json.dumps(new_config, indent=2))

if __name__ == "__main__":
    main()
