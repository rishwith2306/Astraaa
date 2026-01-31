import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Define paths
DATA_DIR = 'Dataset'
LABELS_FILE = os.path.join(DATA_DIR, 'labels.csv')
ANGLES_FILE = os.path.join(DATA_DIR, 'angles.csv')
LANDMARKS_FILE = os.path.join(DATA_DIR, 'landmarks.csv')
DISTANCES_3D_FILE = os.path.join(DATA_DIR, '3d_distances.csv')
XYZ_DISTANCES_FILE = os.path.join(DATA_DIR, 'xyz_distances.csv')
MODEL_FILE = 'exercise_classifier.pkl'

def load_data():
    print("Loading datasets...")
    try:
        labels_df = pd.read_csv(LABELS_FILE)
        angles_df = pd.read_csv(ANGLES_FILE)
        # Optional: Load other features if needed
        # landmarks_df = pd.read_csv(LANDMARKS_FILE)
        # dist_3d_df = pd.read_csv(DISTANCES_3D_FILE)
        # xyz_dist_df = pd.read_csv(XYZ_DISTANCES_FILE)
        
        # Merge datasets
        # Start with labels and angles (usually most critical)
        merged_df = pd.merge(labels_df, angles_df, on='pose_id')
        
        # NOTE: We are intentionally excluding 3d_distances.csv and landmarks.csv (which has Z)
        # because the YOLO model (yolo26n-pose) provides 2D keypoints.
        # Training on 3D data would make the model fail or perform poorly at inference time.
        
        return merged_df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def train():
    df = load_data()
    if df is None:
        return

    print(f"Data loaded. Shape: {df.shape}")
    
    # Separate features and target
    # Assuming 'pose' is the target column in labels.csv
    X = df.drop(columns=['pose_id', 'pose'])
    y = df['pose']
    
    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    print(f"Saving model to {MODEL_FILE}...")
    joblib.dump(clf, MODEL_FILE)
    print("Done.")

if __name__ == "__main__":
    train()
