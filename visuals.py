import cv2
import numpy as np

# COCO Keypoint Skeleton Connections
# Format: (Point A Index, Point B Index)
SKELETON_CONNECTIONS = [
    (5, 7), (7, 9),       # Left Arm
    (6, 8), (8, 10),      # Right Arm
    (11, 13), (13, 15),   # Left Leg
    (12, 14), (14, 16),   # Right Leg
    (5, 6), (11, 12),     # Shoulders, Hips
    (5, 11), (6, 12)      # Torso
]

def draw_skeleton(frame, keypoints, is_correct):
    """
    Draws the skeleton on the frame.
    keypoints: (17, 3) array
    is_correct: bool (Determines color: Green for Good, Red for Bad)
    """
    if keypoints is None:
        return
        
    # BGR Color
    color = (0, 255, 0) if is_correct else (0, 0, 255) 
    
    # Draw Connections (Limbs)
    for p1, p2 in SKELETON_CONNECTIONS:
        # Check if indices are within bounds (just in case)
        if p1 < len(keypoints) and p2 < len(keypoints):
            kp1 = keypoints[p1]
            kp2 = keypoints[p2]
            
            # Check confidence (index 2)
            if kp1[2] > 0.5 and kp2[2] > 0.5:
                pt1 = (int(kp1[0]), int(kp1[1]))
                pt2 = (int(kp2[0]), int(kp2[1]))
                cv2.line(frame, pt1, pt2, color, 3)
            
    # Draw Keypoints (Joints)
    for i, kp in enumerate(keypoints):
        if kp[2] > 0.5:
            cv2.circle(frame, (int(kp[0]), int(kp[1])), 5, color, -1)

def draw_overlay(frame, game):
    """
    Draws the gamified UI: Energy Bar, Score, Reps, Feedback.
    game: Instance of Exercise class (e.g. BicepCurl)
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # --- 1. Feedback Text ---
    # Top Center
    text = str(game.feedback).upper()
    text_color = (0, 255, 0) if game.is_correct_form else (0, 0, 255)
    
    text_scale = 1.0
    thickness = 2
    text_size = cv2.getTextSize(text, font, text_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    cv2.putText(frame, text, (text_x, 50), font, text_scale, text_color, thickness)
    
    # --- 2. Rep Counter & Score ---
    # Top Left
    cv2.putText(frame, f"REPS: {game.reps}", (20, 60), font, 1.2, (255, 255, 0), 3)
    cv2.putText(frame, f"SCORE: {game.score}", (20, 100), font, 0.8, (255, 255, 255), 2)
    
    # Debug: Show current angle
    if hasattr(game, 'current_angle'):
        cv2.putText(frame, f"Ang: {int(game.current_angle)}", (20, 140), font, 0.6, (200, 200, 200), 1)
    
    # --- 3. Energy Bar ---
    # Right side vertical bar
    bar_width = 30
    bar_height = 200
    bar_x = w - 50
    bar_y = 60
    
    # Draw Background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
    
    # Calculate Fill
    # Clamp energy 0-100
    energy_level = max(0, min(100, game.energy))
    fill_height = int((energy_level / 100.0) * bar_height)
    
    # Determine Color
    fill_color = (0, 255, 0) # Green
    if energy_level < 30:
        fill_color = (0, 0, 255) # Red
    elif energy_level < 60:
        fill_color = (0, 165, 255) # Orange
    
    # Draw Fill (Bottom Up)
    start_point = (bar_x, bar_y + bar_height - fill_height)
    end_point = (bar_x + bar_width, bar_y + bar_height)
    cv2.rectangle(frame, start_point, end_point, fill_color, -1)
    
    # Label
    cv2.putText(frame, "HP", (bar_x + 2, bar_y + bar_height + 20), font, 0.6, (255, 255, 255), 1)
