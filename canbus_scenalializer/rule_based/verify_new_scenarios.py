import yaml
import sys
import os

# Add current directory to path so we can import classifier
sys.path.append(os.getcwd())

from classifier import RuleBasedClassifier

def verify_scenarios():
    config_path = 'config.yaml'
    classifier = RuleBasedClassifier(config_path)
    
    # Define test cases
    test_cases = [
        {
            "name": "U-Turn Test",
            "frame": {
                "speed": 5.0,
                "steering_angle": 5.0, # > 4.327
                "yaw_rate": 0.1,
                "turn_signal": 1
            },
            "expected": "U-Turn"
        },
        {
            "name": "Pull Over Test",
            "frame": {
                "speed": 5.0, # < 11.373
                "steering_angle": 0.3, # > 0.282
                "yaw_rate": 0.05,
                "turn_signal": 1 # != 0
            },
            "expected": "Pull Over"
        },
        {
            "name": "Left Turn Test (Regression)",
            "frame": {
                "speed": 5.0,
                "steering_angle": 1.5, # > 1.013 but < 4.327
                "yaw_rate": 0.1,
                "turn_signal": 1
            },
            "expected": "Left Turn"
        },
        {
            "name": "Lane Change Test (Regression)",
            "frame": {
                "speed": 12.0, # > 11.373 (Cruising speed)
                "steering_angle": 0.3, # > 0.282
                "yaw_rate": 0.05,
                "turn_signal": 1
            },
            "expected": "Lane Change"
        }
    ]
    
    all_passed = True
    print("Running verification tests...")
    for case in test_cases:
        result = classifier._classify_frame(case['frame'])
        if result == case['expected']:
            print(f"[PASS] {case['name']}: Expected '{case['expected']}', Got '{result}'")
        else:
            print(f"[FAIL] {case['name']}: Expected '{case['expected']}', Got '{result}'")
            all_passed = False
            
    if all_passed:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed.")

if __name__ == "__main__":
    verify_scenarios()
