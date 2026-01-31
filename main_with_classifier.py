import cv2
import time
import sys
import numpy as np
import joblib
import pandas as pd

# Import local modules
try:
    from pose_engine import PoseEngine
    from game_logic import ClassifierExercise
    import visuals
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

CLASSIFIER_MODEL = 'exercise_classifier.pkl'

def calculate_angle_2d(a, b, c):
    """
    Calculates angle 2D similarly to how we assume angles.csv was generated.
    Points are (x, y)
    """
    a = np.array(a[:2])
    b = np.array(b[:2])
    c = np.array(c[:2])
    
    ba = a - b
    bc = c - b
    
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
        
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def extract_features(keypoints):
    """
    Extracts the EXACT 7 features used in training from YOLO keypoints.
    KEYPOINTS are (17, 3) --> [x, y, conf]
    """
    # Keypoint Map (COCO)
    # 0:Nose, 1:LEye, 2:REye, 3:LEar, 4:REar, 5:LShoulder, 6:RShoulder
    # 7:LElbow, 8:RElbow, 9:LWrist, 10:RWrist, 11:LHip, 12:RHip
    # 13:LKnee, 14:RKnee, 15:LAnkle, 16:RAnkle
    
    kp = keypoints # Access shortcut
    
    # 1. right_elbow_right_shoulder_right_hip
    feat1 = calculate_angle_2d(kp[8], kp[6], kp[12])
    
    # 2. left_elbow_left_shoulder_left_hip
    feat2 = calculate_angle_2d(kp[7], kp[5], kp[11])
    
    # 3. right_knee_mid_hip_left_knee
    mid_hip = (kp[11][:2] + kp[12][:2]) / 2
    feat3 = calculate_angle_2d(kp[14], mid_hip, kp[13])
    
    # 4. right_hip_right_knee_right_ankle
    feat4 = calculate_angle_2d(kp[12], kp[14], kp[16])
    
    # 5. left_hip_left_knee_left_ankle
    feat5 = calculate_angle_2d(kp[11], kp[13], kp[15])
    
    # 6. right_wrist_right_elbow_right_shoulder
    feat6 = calculate_angle_2d(kp[10], kp[8], kp[6])
    
    # 7. left_wrist_left_elbow_left_shoulder
    feat7 = calculate_angle_2d(kp[9], kp[7], kp[5])
    
    features = pd.DataFrame([{
        'right_elbow_right_shoulder_right_hip': feat1,
        'left_elbow_left_shoulder_left_hip': feat2,
        'right_knee_mid_hip_left_knee': feat3,
        'right_hip_right_knee_right_ankle': feat4,
        'left_hip_left_knee_left_ankle': feat5,
        'right_wrist_right_elbow_right_shoulder': feat6,
        'left_wrist_left_elbow_left_shoulder': feat7
    }])
    
    return features


def main():
    print("--- PROJECT: GAMIFIED POSE TRACKER (W/ CLASSIFIER) ---")
    print(f"Loading Classifier: {CLASSIFIER_MODEL}...")
    try:
        clf = joblib.load(CLASSIFIER_MODEL)
        print("Classifier loaded successfully.")
    except Exception as e:
        print(f"Could not load classifier: {e}")
        return

    try:
        engine = PoseEngine(model_path="yolo26n-pose.pt", device=0)
    except Exception as e:
        print("CRITICAL ERROR: Could not load PoseEngineModel.")
        return

    # Initialize Games for each supported exercise
    # These persist reps/score across the session
    games = {
        'squats': ClassifierExercise('squats'),
        'pushups': ClassifierExercise('pushups'),
        'jumping_jacks': ClassifierExercise('jumping_jacks'),
        'pullups': ClassifierExercise('pullups'),
        'situp': ClassifierExercise('situp')
    }
    
    # active_game will point to one of the instances in 'games'
    active_game = None 
    active_exercise_name = "None"
    
    # Open Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not found.")
        return

    print("Starting Main Loop. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        keypoints = engine.get_keypoints(frame)
        
        # If person detected
        if keypoints is not None:
            # 1. Visualize Skeleton
            visuals.draw_skeleton(frame, keypoints, is_correct=True)
            
            # 2. Key Logic
            try:
                # A. Extract Features
                feats = extract_features(keypoints)
                
                # B. Predict Exercise
                pred_label = clf.predict(feats)[0] # e.g. "squats_down"
                probs = clf.predict_proba(feats)[0]
                confidence = np.max(probs)
                
                # C. Determine Exercise Type (e.g. "squats" from "squats_down")
                exercise_type = pred_label.rsplit('_', 1)[0] # "squats_down" -> "squats"
                
                # D. Switch Game Mode if needed
                # Only switch if high confidence and different from current
                if confidence > 0.6:
                    if exercise_type in games:
                        active_game = games[exercise_type]
                        active_exercise_name = exercise_type
                
                # E. Update Active Game
                if active_game:
                    active_game.update(pred_label, confidence)
                    
                    # F. Draw Game Overlay
                    visuals.draw_overlay(frame, active_game)
                    
                    # Draw Active Exercise Label manually if not in overlay
                    cv2.putText(frame, f"MODE: {active_exercise_name.upper()}", (20, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    
            except Exception as e:
                # print(f"Logic Error: {e}")
                pass
            
        else:
            cv2.putText(frame, "No Person Detected", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show Frame
        cv2.imshow("Astraa Tracker - Gamified", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
