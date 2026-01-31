import cv2
import time
import sys

# Import local modules
try:
    from pose_engine import PoseEngine
    from game_logic import BicepCurl
    import visuals
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def main():
    print("--- PROJECT: GAMIFIED POSE TRACKER ---")
    
    try:
        engine = PoseEngine(model_path="yolo26n-pose.pt", device=0)
    except Exception as e:
        print("CRITICAL ERROR: Could not load PoseEngineModel.")
        print(f"Details: {e}")
        return    
    game = BicepCurl()
    print("Opening Webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Game Started! Stand back and perform a Bicep Curl.")
    print("Press 'q' to Quit.")
    prev_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            time.sleep(1)
            continue

        
        frame = cv2.flip(frame, 1)

        
        keypoints = engine.get_keypoints(frame)

        
        if keypoints is not None:
            game.update(keypoints)
        else:
            
            game.feedback = "Looking for Player..."
            
            
        # 6. Visualization
        visuals.draw_skeleton(frame, keypoints, game.is_correct_form)
        visuals.draw_overlay(frame, game)

        # FPS Counter
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 7. Render
        cv2.imshow('GAMIFIED POSE TRACKER', frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Game Exited.")

if __name__ == "__main__":
    main()
