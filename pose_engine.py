import torch
import numpy as np
from ultralytics import YOLO

class PoseEngine:
    """
    Wrapper class for YOLO26-Pose loading and inference.
    """
    def __init__(self, model_path="yolo26n-pose.pt", device=0):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # If user insisted on a specific device index:
        if isinstance(device, int) and 'cuda' in self.device:
            self.device = f'cuda:{device}'
        
        print(f"[PoseEngine] Loading {model_path} on {self.device}...")
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
        except Exception as e:
            print(f"[PoseEngine] Error loading model: {e}")
            print("Please ensure 'yolo26n-pose.pt' exists or use a standard YOLOv8 pose model like 'yolov8n-pose.pt'.")
            raise e

    def get_keypoints(self, frame):
        """
        Runs inference on the frame and returns keypoints for the primary person.
        Args:
            frame (np.ndarray): The BGR image frame from OpenCV.
        Returns:
            np.ndarray: Shape (17, 3) containing (x, y, conf) for each keypoint.
                        Returns None if no person is detected.
        """
        # Run inference
        # verbose=False for speed
        results = self.model(frame, verbose=False, stream=False)
        
        if not results:
            return None

        # Take the first result
        result = results[0]
        
        # Check if keypoints exist
        if result.keypoints is not None and result.keypoints.data is not None and result.keypoints.data.shape[1] > 0:
            # result.keypoints.data is shape (N, 17, 3) (batch, kpts, xyconf)
            # We assume the first detection is the primary player
            if len(result.keypoints.data) > 0:
                # Return detection 0, all keypoints, all vals (x,y,conf)
                # Move to CPU and numpy
                kp = result.keypoints.data[0].cpu().numpy()
                return kp
        
        return None
