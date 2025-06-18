import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import csv
import os
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('capture_log.txt', mode='w')
    ]
)
logger = logging.getLogger(__name__)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logger.error("Could not open webcam")
    exit()

csv_path = 'data/test.csv'
image_dir = 'data/images'
os.makedirs(image_dir, exist_ok=True)
os.makedirs(os.path.dirname(csv_path), exist_ok=True)

class_labels = {0: 'Open', 1: 'Close', 2: 'Pointer', 3: 'Victory'}
capture_trigger = False
class_id = None

def pre_process_landmark(landmark_list):
    temp_landmark_list = landmark_list.copy()
    base_x, base_y = temp_landmark_list[0], temp_landmark_list[1]
    for i in range(0, len(temp_landmark_list), 2):
        temp_landmark_list[i] -= base_x
        temp_landmark_list[i + 1] -= base_y
    max_value = max(abs(x) for x in temp_landmark_list) if any(temp_landmark_list) else 1
    temp_landmark_list = [x / max_value for x in temp_landmark_list]
    return temp_landmark_list

def save_data(class_id, image, hand_landmarks):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    image_path = os.path.join(image_dir, f'gesture_{class_labels.get(class_id, "unknown")}_{timestamp}.png')
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(image_path)
    
    kp_list = []
    for lm in hand_landmarks.landmark:
        x, y = lm.x, lm.y
        kp_list.extend([x, y])
    if len(kp_list) != 42:
        logger.error(f"Expected 42 keypoints, got {len(kp_list)} for class {class_labels.get(class_id, 'unknown')}")
        return
    kp_normalized = pre_process_landmark(kp_list)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([class_id] + kp_normalized)
    logger.info(f"Saved {class_labels.get(class_id, 'unknown')} sample to {csv_path}, image: {image_path}")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to capture frame")
            continue
        frame_bgr = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                if capture_trigger and class_id is not None:
                    if results.multi_hand_landmarks:
                        save_data(class_id, frame_bgr, hand_landmarks)
                        capture_trigger = False
                    else:
                        logger.warning(f"No hand detected, skipping capture for class {class_labels.get(class_id, 'unknown')}")
                        capture_trigger = False
        
        label = class_labels.get(class_id, "None")
        cv2.putText(frame_bgr, f'Class: {label} ({class_id})', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Hand Gesture Capture', frame_bgr)
        
        key = cv2.waitKey(10) & 0xFF
        if key == ord('s'):
            capture_trigger = True
        elif key in [ord(str(i)) for i in range(4)]:
            class_id = int(chr(key))
        elif key == ord('q'):
            break
except KeyboardInterrupt:
    logger.info("Capture interrupted")
finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    logger.info("Cleaned up resources")