import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2
import sys
from pathlib import Path

# Add tools directory to path to import inference
sys.path.append(str(Path(__file__).parent.parent / "tools"))

from inference import RoadMarkingDetector, CLASSES

class TestRoadMarkingDetector(unittest.TestCase):
    def setUp(self):
        # Mock MMSegInferencer to avoid loading model
        with patch('inference.MMSegInferencer') as mock_inferencer:
            self.detector = RoadMarkingDetector("dummy_config", "dummy_checkpoint", device='cpu')
            self.detector.inferencer = MagicMock()

    def test_extract_instances(self):
        # Create a dummy mask (100x100)
        mask = np.zeros((100, 100), dtype=np.uint8)
        
        # Draw a square for class 1 (SOLID)
        # x=10, y=10, w=20, h=20
        mask[10:30, 10:30] = 1
        
        instances = self.detector.extract_instances(mask)
        
        self.assertEqual(len(instances), 1)
        inst = instances[0]
        self.assertEqual(inst['class_id'], 1)
        self.assertEqual(inst['label'], 'solid')
        
        # Check bbox [x, y, w, h]
        # Note: cv2.boundingRect might be slightly different depending on implementation, but should be close
        bbox = inst['bbox']
        self.assertEqual(bbox, [10, 10, 20, 20])
        
        # Check polygon points (should be non-empty)
        self.assertTrue(len(inst['polygon']) > 0)

    def test_extract_instances_multiple_classes(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Class 1 at top left
        mask[10:30, 10:30] = 1
        # Class 2 at bottom right
        mask[70:90, 70:90] = 2
        
        instances = self.detector.extract_instances(mask)
        self.assertEqual(len(instances), 2)
        
        class_ids = sorted([inst['class_id'] for inst in instances])
        self.assertEqual(class_ids, [1, 2])

    def test_process_image_flow(self):
        # Mock inferencer return value
        dummy_mask = np.zeros((512, 1024), dtype=np.uint8)
        self.detector.inferencer.return_value = {'predictions': dummy_mask}
        
        result = self.detector.process_image("dummy_path.jpg")
        
        self.assertIn('predictions', result)
        self.detector.inferencer.assert_called_once()

if __name__ == '__main__':
    unittest.main()
