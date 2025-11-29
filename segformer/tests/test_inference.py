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

    def test_process_batch_flow(self):
        # Mock inferencer return value for batch
        dummy_mask1 = np.zeros((512, 1024), dtype=np.uint8)
        dummy_mask2 = np.zeros((512, 1024), dtype=np.uint8)
        
        # MMSegInferencer returns dict with list of predictions
        self.detector.inferencer.return_value = {'predictions': [dummy_mask1, dummy_mask2]}
        
        results = self.detector.process_batch(["img1.jpg", "img2.jpg"], batch_size=2)
        
        self.assertEqual(len(results), 2)
        self.assertIn('predictions', results[0])
        self.assertIn('predictions', results[1])
        self.detector.inferencer.assert_called_once()
        
        # Verify arguments passed to inferencer
        args, kwargs = self.detector.inferencer.call_args
        self.assertEqual(args[0], ["img1.jpg", "img2.jpg"])
        self.assertEqual(kwargs['batch_size'], 2)

if __name__ == '__main__':
    unittest.main()
