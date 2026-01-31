import numpy as np

class Exercise:
    def __init__(self):
        self.reps = 0
        self.score = 0
        self.state = "extension" 
        self.is_correct_form = True
        self.feedback = "Get Ready"
        self.energy = 100.0
        
    def calculate_angle(self, a, b, c):
        """
        Calculates the angle at vertex b formed by points a and c.
        points are [x, y, confidence] or just [x, y] arrays.
        """
        # Take only x,y
        a = np.array(a[:2])
        b = np.array(b[:2])
        c = np.array(c[:2])
        
        ba = a - b
        bc = c - b
        
        # Norms
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        if norm_ba == 0 or norm_bc == 0:
            return 0.0
            
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        # Clip to handle floating point errors
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))
        
        return angle

class BicepCurl(Exercise):
    def __init__(self):
        super().__init__()
        # COCO Keypoint Indices
        self.KP_R_SHOULDER = 6
        self.KP_R_ELBOW = 8
        self.KP_R_WRIST = 10
        self.KP_R_HIP = 12
        
        self.current_angle = 0.0
        self.flare_angle = 0.0

    def update(self, keypoints):
        """
        Process pose keypoints and update game state.
        keypoints: (17, 3) numpy array
        """
        # 1. Check if required points are detected
        needed = [self.KP_R_SHOULDER, self.KP_R_ELBOW, self.KP_R_WRIST, self.KP_R_HIP]
        for idx in needed:
            if keypoints[idx][2] < 0.5: # Low confidence
                self.feedback = "Camera Obstructed"
                self.is_correct_form = False
                return

        # Extract Points
        s = keypoints[self.KP_R_SHOULDER]
        e = keypoints[self.KP_R_ELBOW]
        w = keypoints[self.KP_R_WRIST]
        h = keypoints[self.KP_R_HIP]

        # 2. Analyze Form (Elbow Flare)
        # Angle at Shoulder (S) between Hip (H) and Elbow (E)
        self.flare_angle = self.calculate_angle(h, s, e)
        
        # Valid flare is usually < 20-30 degrees
        if self.flare_angle > 20: 
            self.is_correct_form = False
            self.feedback = "Tuck Your Elbow!"
            self.energy -= 0.5 # Drain energy
        else:
            self.is_correct_form = True
            self.feedback = "Good Form"
            self.energy = min(100, self.energy + 0.2) # Regen

        # 3. Analyze Repetition (Flexion/Extension)
        # Angle at Elbow (E) between Shoulder (S) and Wrist (W)
        self.current_angle = self.calculate_angle(s, e, w)

        # State Machine
        if self.state == "extension":
            # Waiting to curl up
            if self.current_angle < 40: # Fully curled
                # Only count rep if form was good recently? 
                # For MVP, just check current form
                if self.is_correct_form:
                    self.state = "flexion"
                    self.reps += 1
                    self.score += 100
                    # Trigger visual pop or sound in main
        
        elif self.state == "flexion":
            # Waiting to extend down
            if self.current_angle > 160: # Fully extended
                self.state = "extension"

        # Energy clamp
        if self.energy < 0:
            self.energy = 0
            self.feedback = "FATIGUE / BAD FORM"


class ClassifierExercise(Exercise):
    """
    Generic exercise class that relies on external Classifier predictions 
    (e.g., 'squats_down', 'squats_up') rather than manual angle calculations.
    """
    def __init__(self, exercise_name="squats"):
        super().__init__()
        self.exercise_name = exercise_name # e.g. "squats", "pushups", "jumping_jacks"
        self.last_pred = ""
        self.state = "start"
        
        # Define what constitutes a "rep" for this exercise
        # Usually going from 'down' to 'up' or vice-versa
        self.target_state = f"{exercise_name}_down" # The "active" part
        self.reset_state = f"{exercise_name}_up"    # The "rest" part
        
        # Adjust for exercises where logic might be inverted or named differently
        if exercise_name == "pullups":
            self.target_state = "pullups_up"
            self.reset_state = "pullups_down"

    def update(self, prediction, confidence):
        """
        prediction: str (e.g. 'squats_down')
        confidence: float
        """
        if confidence < 0.4:
            self.feedback = "Uncertain..."
            return

        # Basic filtering to avoid flickering
        if prediction == self.last_pred:
            pass # Stable state
        else:
            # State transition detection
            self.handle_transition(prediction)
            self.last_pred = prediction

        # Form feedback is implicit in the classifier's ability to detect it at all?
        # Or we can say if it's "Uncertain" often, form is bad.
        # For now, if we match the target/reset states, we assume Good Form.
        if prediction in [self.target_state, self.reset_state]:
            self.is_correct_form = True
            self.feedback = f"Doing {self.exercise_name}..."
            self.energy = min(100, self.energy + 0.1)
        else:
            # Maybe detecting "standing" or other exercises
            pass

    def handle_transition(self, new_state):
        # State Machine
        # simple 2-state: RECOVER -> ACTIVE -> RECOVER (Count Rep)
        
        # If we were in reset state and moved to target state
        if new_state == self.target_state and self.state == "rest":
            self.state = "active"
            self.feedback = "GO!"
            
        # If we were active and moved back to reset/rest state -> REP COMPLETE
        elif new_state == self.reset_state and self.state == "active":
            self.state = "rest"
            self.reps += 1
            self.score += 100
            self.feedback = "GOOD REP!"
            self.energy = min(100, self.energy + 5)
            
        # Initialization
        elif self.state == "start":
            if new_state == self.reset_state:
                self.state = "rest"
            elif new_state == self.target_state:
                self.state = "active"
