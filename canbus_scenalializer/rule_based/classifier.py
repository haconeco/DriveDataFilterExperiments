import yaml
import json
import numpy as np

class RuleBasedClassifier:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            self.config = json.load(f)
        self.thresholds = self.config['thresholds']
    def classify(self, can_data):
        """
        Classify a sequence of CAN data.
        
        Args:
            can_data (list of dict): List of CAN bus data points. 
                                     Expected keys: 'speed', 'steering_angle', 'yaw_rate', 'turn_signal'
        
        Returns:
            list of str: List of scenario labels corresponding to each data point.
        """
        results = []
        for frame in can_data:
            results.append(self._classify_frame(frame))
        return results

    def _classify_frame(self, frame):
        """
        Classify a single frame of CAN data.
        
        Args:
            frame (dict): Single CAN bus data point.
        
        Returns:
            str: Scenario label.
        """
        speed = frame.get('speed', 0.0)
        steering = abs(frame.get('steering_angle', 0.0))
        yaw_rate = abs(frame.get('yaw_rate', 0.0))
        turn_signal = frame.get('turn_signal', 0) # 0: None, 1: Left, 2: Right (Example encoding)
        
        # 1. Stop
        if speed < self.thresholds['stop_speed_threshold']:
            return "Stop"
        
        # 2. Reverse (Assuming negative speed or gear info is available, here using simple speed check if signed)
        # Note: NuScenes speed is often magnitude, so we might need gear info. 
        # For now, if speed is negative (some datasets) or gear is 'R'.
        if frame.get('gear') == 'R':
             return "Reverse"

        # 3. U-Turn (Very high steering angle)
        if steering > self.thresholds['u_turn_steering_threshold']:
             return "U-Turn"

        # 4. Turn (High steering angle or high yaw rate)
        if steering > self.thresholds['turn_steering_threshold']:
            # Determine Left or Right based on sign if available, or signal
            # Assuming steering_angle > 0 is Left (standard in many ISO), but need to verify dataset specific.
            # NuScenes: positive is left.
            if frame.get('steering_angle', 0.0) > 0:
                return "Left Turn"
            else:
                return "Right Turn"

        # 5. Pull Over (Signal + Steering + Low Speed)
        if turn_signal != 0 and steering > self.thresholds['lane_change_steering_threshold'] and speed < self.thresholds['pull_over_speed_threshold']:
             return "Pull Over"

        # 6. Lane Change (Moderate steering + Turn Signal)
        # Note: Lane change is hard to distinguish from curve without map, but signal is a strong cue.
        if turn_signal != 0 and steering > self.thresholds['lane_change_steering_threshold']:
             return "Lane Change"

        # 5. Deceleration (Need previous frame or acceleration field, simplified here)
        accel = frame.get('acceleration', 0.0)
        if accel < -1.0: # Threshold for significant deceleration
            return "Deceleration"

        # 6. Cruising (Default)
        return "Cruising"
