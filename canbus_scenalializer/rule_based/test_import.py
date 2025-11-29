import sys
print("Python executable:", sys.executable)
try:
    import nuscenes
    print("nuscenes imported successfully")
    from nuscenes.nuscenes import NuScenes
    print("NuScenes class imported")
    import cv2
    print("cv2 imported successfully")
    import yaml
    print("yaml imported successfully")
except Exception as e:
    print(f"Import error: {e}")
