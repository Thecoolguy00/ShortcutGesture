# Hand Gesture Recognition with MediaPipe

This project implements a hand gesture recognition system using MediaPipe and a TFLite model to detect gestures like "Open," "Close," "Pointer," and "Victory." It includes scripts for capturing data, training a model, and performing real-time inference to minimize/restore apps on Windows based on gesture transitions.

This project is just the base version like a game in it's beta phase, I plan to add more shortcuts, dynamic gesture recognition and 2 hand-mode

This repository contains the following contents
* Sample program
* Hand gesture recognition model(TFLite)
* Data for training hand gesture recognition model
* Various Scripts which are explained down below
* Requirements list

## Requirements
- Python 3.8.10
- Windows OS (due to `pyautogui` usage for triggering windows shorcuts)

## Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/hand-gesture-recognition.git
   cd hand-gesture-recognition
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv shortcut_gesture
   shortcut_gesture\Scripts\activate  # On Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure the `model/gesture_classifier.tflite` file is in the `model/` directory (required).

# Directory
<pre>
│  hand_gesture_shortcut.py
│  infer.py
│  capture.py
│
├─model
│  │ gesture_classifier.hdf5
│  └─gesture_classifier.tflite
│
└─data
    └─keypoint.csv
</pre>
## Scripts

### 1. `hand_gesture_shortcut.py`
**Purpose**: Performs real-time gesture recognition to minimize/restore apps.  
**Usage**:
   ```bash
   python hand_gesture_shortcut.py
   ```
- Uses the webcam to detect hand gestures.
- Perform an "Open → Close" gesture sequence to minimize all apps.
- Perform a "Close → Open" gesture sequence to restore apps.
- Press `q` to quit the application.

### 2. `capture.py`
**Purpose**: Captures hand gesture data using the webcam and saves it for training.  
**Usage**:
   ```bash
   python capture.py
   ```
- Captures video frames and detects hand landmarks using MediaPipe.
- Saves keypoints or images to files (e.g., CSV or image files) for training.
- Check the script for specific output paths and formats.

### 3. `train.py`
**Purpose**: Trains a TFLite model using captured data.  
**Usage**:
   ```bash
   python train.py
   ```
- Loads captured data (e.g., keypoints from `capture.py`).
- Trains a model using TensorFlow and converts it to TFLite format (`model/gesture_classifier.tflite`).
- Ensure captured data is available before running.

### 4. `infer.py`
**Purpose**: for testing gesture recognition.
**Usage**:
   ```bash
   python infer.py
   ```
- Uses the webcam to detect hand gestures.
- Press `q` to quit the application.

## Notes
- Logs are saved to `gesture_shortcut_log.txt` for `hand_gesture_shortcut.py` and `infer_log.txt` for `infer.py`.
- Debug images are saved to the `debug_frames/` directory when confidence is between 0.4 and 0.5 (every 30 frames).
- Ensure your webcam is working and lighting conditions are good for accurate gesture detection.

## Inspiration and Credits
This project was inspired by the following sources:
- **YouTube Video by Ivan Goncharov**: I got the motivation to start this project from this video [Ivan Goncharov](https://youtu.be/a99p_fAr6e4?si=0DLq0tjst_LVRbaP), which demonstrated hand gesture recognition using MediaPipe and clearly explained the whole working of the model.
- **Kazuhito's Hand-Gesture MediaPipe Line**: The gesture detection pipeline in this project takes reference from [Kazuhito's hand-gesture recognition work](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe), which provided a solid foundation for integrating MediaPipe with gesture recognition. This one has way more features than mine, the only reason that I didn't fork it is that I was having many problems with it's dependencies and especially with training a model for new gestures.

# Reference
* [MediaPipe](https://mediapipe.dev/)

# Author
Arvindraj Ramesh(https://x.com/Arvind_224)
 
# License 
ShortcutGesture is under [Apache v2 license](LICENSE).