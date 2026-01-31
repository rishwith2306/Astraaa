# Astraa: Gamified Pose Tracker

A real-time AI computer vision application that tracks your body movements using a webcam, classifies exercises, and gamifies your workout experience.

## Features

-   **Real-time Pose Estimation**: Uses YOLOv8/YOLO11-Pose models to track keypoints (joints) on your body.
-   **Exercise Classification**: Automatically detects what exercise you are performing (Squats, Jumping Jacks, Pushups, etc.).
-   **Rep Counting**: Intelligently counts repetitions based on form analysis.
-   **Gamification**: Scores points, tracks "Health/Energy" based on form, and provides visual feedback.
-   **Cross-Platform**: Runs on Windows, macOS, and Linux.

## Supported Exercises

The system is trained to recognize and count reps for:

1.  **Squats**
2.  **Jumping Jacks**
3.  **Pushups**
4.  **Pullups**
5.  **Situps**

## Quick Start

### Windows
Double-click `run_windows.bat`. 

Or run from command line:
```cmd
run_windows.bat
```

### macOS / Linux
Open a terminal, give the script permission, and run it:
```bash
chmod +x run_unix.sh
./run_unix.sh
```

## Manual Installation

If you prefer to set up manually:

1.  **Helper**: Install Python 3.8 or higher.
2.  **Clone/Download** this repository.
3.  **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
4.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
5.  **Run the App**:
    ```bash
    python main_with_classifier.py
    ```

## Training Custom Models

If you want to retrain the classifier on your own dataset:

1.  Place your dataset CSVs (`angles.csv`, `labels.csv`) in the `Dataset/` folder.
2.  Run the training script:
    ```bash
    python train_model.py
    ```
3.  This will generate a new `exercise_classifier.pkl` file which usage `main_with_classifier.py`.

## Controls
-   **Q**: Quit the application.

## Troubleshooting

-   **Camera not found**: Ensure your webcam is plugged in and not being used by another application (like Zoom or Teams).
-   **Slow Performance**: If running on CPU, performance depends on your processor speed. For faster performance, ensure you have an NVIDIA GPU and install the CUDA version of PyTorch manually before running requirements.
