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

def fit_gmm_and_find_threshold(data, n_components, threshold_percentile=99, lower_bound=True):
    # Reshape for sklearn
    X = data.reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X)
    
    # Sort components by mean
    means = gmm.means_.flatten()
    sorted_indices = np.argsort(means)
    
    # For speed: Component 0 is likely "Stop", Component 1+ is "Move"
    # For steering: Component 0 is likely "Straight", Component 1+ is "Turn"
    
    # We want to find a boundary.
    # Simple approach: Take the dominant "low value" component (Stop/Straight) 
    # and find the value where its PDF drops or a specific percentile.
    
    low_component_idx = sorted_indices[0]
    mean = means[low_component_idx]
    cov = gmm.covariances_.flatten()[low_component_idx]
    std = np.sqrt(cov)
    
    # Suggest threshold as mean + 3*std (approx 99.7% of the component)
    threshold = mean + 3 * std
    
    return threshold, gmm

def plot_distribution(data, gmm, threshold, title, xlabel, filename):
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

    # Plot threshold
    plt.axvline(threshold, color='r', linestyle='dashed', linewidth=2, label=f'Threshold: {threshold:.3f}')
    
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
    
    # 1. Speed Threshold (Stop vs Move)
    # Fit 2 components: Stop (near 0) and Move
    print("Fitting Speed GMM...")
    stop_speed_threshold, speed_gmm = fit_gmm_and_find_threshold(speeds, n_components=2)
    plot_distribution(speeds, speed_gmm, stop_speed_threshold, 'Speed Distribution', 'Speed (m/s)', os.path.join(output_dir, 'dist_speed.png'))
    
    # 2. Steering Threshold (Straight vs Turn)
    # Fit 2 components: Straight (near 0) and Turn
    print("Fitting Steering GMM...")
    turn_steering_threshold, steer_gmm = fit_gmm_and_find_threshold(steerings, n_components=2)
    plot_distribution(steerings, steer_gmm, turn_steering_threshold, 'Steering Angle Distribution', 'Steering Angle (rad)', os.path.join(output_dir, 'dist_steering.png'))
    
    # Lane change is subtle, usually smaller than turn. 
    # Maybe we can define lane change as something between "Strictly Straight" and "Turn".
    # For now, let's use a fraction of the turn threshold or a separate analysis if we had labels.
    # Let's heuristically set lane_change as 1/3 of turn threshold or 1 sigma?
    # Let's use 1 sigma of the straight component for lane change start?
    # Actually, let's stick to the GMM result. The "Straight" component width defines "Cruising".
    # Anything outside "Straight" but not yet "Turn" could be Lane Change?
    # Let's set lane_change_threshold as mean + 1*std of straight component.
    
    means = steer_gmm.means_.flatten()
    sorted_indices = np.argsort(means)
    straight_idx = sorted_indices[0]
    straight_std = np.sqrt(steer_gmm.covariances_.flatten()[straight_idx])
    lane_change_steering_threshold = means[straight_idx] + 1.5 * straight_std # 1.5 sigma
    
    # 3. Yaw Rate Threshold
    print("Fitting Yaw Rate GMM...")
    yaw_rate_threshold, yaw_gmm = fit_gmm_and_find_threshold(yaw_rates, n_components=2)
    plot_distribution(yaw_rates, yaw_gmm, yaw_rate_threshold, 'Yaw Rate Distribution', 'Yaw Rate (rad/s)', os.path.join(output_dir, 'dist_yaw.png'))
    
    # Generate Config
    new_config = {
        "thresholds": {
            "stop_speed_threshold": float(stop_speed_threshold),
            "turn_steering_threshold": float(turn_steering_threshold),
            "lane_change_steering_threshold": float(lane_change_steering_threshold),
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
