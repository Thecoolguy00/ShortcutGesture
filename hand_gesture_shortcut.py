import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import logging
import os
import time
import pyautogui
from collections import deque

# Configure pyautogui
pyautogui.FAILSAFE = False  # Disable failsafe (moving mouse to corner won't stop the script)
pyautogui.PAUSE = 0.01     # Small pause between actions for stability

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gesture_shortcut_log.txt', mode='w')
    ]
)
logger = logging.getLogger(__name__)

def pre_process_landmark(landmark_list):
    temp_landmark_list = landmark_list.copy()
    base_x, base_y = temp_landmark_list[0], temp_landmark_list[1]
    for i in range(0, len(temp_landmark_list), 2):
        temp_landmark_list[i] -= base_x
        temp_landmark_list[i + 1] -= base_y
    max_value = max(abs(x) for x in temp_landmark_list) if any(temp_landmark_list) else 1
    temp_landmark_list = [x / max_value for x in temp_landmark_list]
    return temp_landmark_list

# Initialize MediaPipe
try:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    logger.info("Initialized MediaPipe Hands")
except Exception as e:
    logger.error(f"Failed to initialize MediaPipe: {e}")
    exit()

# Load TFLite model
model_path = 'model/victory.tflite'
try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.info(f"Model input shape: {input_details[0]['shape']}")
    logger.info(f"Model output shape: {output_details[0]['shape']}")
except Exception as e:
    logger.error(f"Failed to load TFLite model: {e}")
    exit()

labels = ['Open', 'Close', 'Pointer', 'Victory']

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logger.error("Could not open webcam")
    exit()

# Gesture sequence tracking
FRAME_WINDOW = 10  # Number of frames to track for sequence (e.g., ~0.33s at 30 FPS)
gesture_buffer = deque(maxlen=FRAME_WINDOW)  # Buffer to store recent gesture labels
last_action_time = 0  # Timestamp of last minimize action
COOLDOWN_SECONDS = 0.5  # Cooldown period to prevent repeated triggers, reduced to 0.5 from 2.0 for a more quick responsiveness
fps=0
fps_frame_count=0
fps_start_time=time.time()

frame_count = 0
os.makedirs('debug_frames', exist_ok=True)

while True:
    try:
        ret, frame = cap.read()
        frame_count += 1

        #fps counter
        fps_frame_count+=1
        current_time=time.time()
        elapsed_time=current_time-fps_start_time
        if elapsed_time>=1.0:
            fps=fps_frame_count/elapsed_time
            logger.info(f"FPS: {fps:.2f}")
            fps_frame_count=0
            fps_start_time=current_time

        if not ret:
            logger.warning(f"Failed to read frame {frame_count}")
            continue

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        keypoints = []
        predicted_label = None
        confidence = 0.0
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            raw_keypoints = []
            for i in range(21):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                raw_keypoints.extend([x, y])
            raw_keypoints = np.array(raw_keypoints, dtype=np.float32)
            #logger.debug(f"Raw keypoints: min={raw_keypoints.min():.3f}, max={raw_keypoints.max():.3f}")

            keypoints = pre_process_landmark(raw_keypoints)
            keypoints = np.array(keypoints, dtype=np.float32)
            #logger.debug(f"Normalized keypoints: min={keypoints.min():.3f}, max={keypoints.max():.3f}")

            keypoints_np = keypoints.reshape(1, 42)
            interpreter.set_tensor(input_details[0]['index'], keypoints_np)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_class = np.argmax(output_data[0])
            predicted_label = labels[predicted_class]
            confidence = output_data[0][predicted_class]
            confidences = {labels[i]: f"{output_data[0][i]:.2f}" for i in range(len(labels))}

            # logger.info(f"Frame {frame_count}: {predicted_label}, confidence={confidence:.2f}, {confidences}")

            # Add the predicted label to the buffer (only if confidence is high)
            if confidence >= 0.6:
                gesture_buffer.append(predicted_label)
            else:
                logger.debug(f"Low confidence for {predicted_label}: {confidence:.2f}")
                logger.info(f"Frame {frame_count}: {predicted_label}, confidence={confidence:.2f}, {confidences}")
                gesture_buffer.append(None)  # Avoid using low-confidence predictions

            # Check for "Open -> Close" and "Close -> Open" transitions
            current_time = time.time()
            if len(gesture_buffer) == FRAME_WINDOW and (current_time - last_action_time) >= COOLDOWN_SECONDS:

                # Look for "Open" followed by "Close" in the buffer
                has_open = False
                for label in list(gesture_buffer)[:FRAME_WINDOW//2]:  # First half of buffer
                    if label == 'Open':
                        logger.info(f"Frame {frame_count}: {predicted_label}, confidence={confidence:.2f}, {confidences}")
                        has_open = True
                        break
                has_close = False
                for label in list(gesture_buffer)[FRAME_WINDOW//2:]:  # Second half of buffer
                    if label == 'Close':
                        logger.info(f"Frame {frame_count}: {predicted_label}, confidence={confidence:.2f}, {confidences}")
                        has_close = True
                        break

                if has_open and has_close:
                    logger.debug(f"Gesture buffer content: {list(gesture_buffer)}")
                    logger.info("Detected Open -> Close transition, minimizing all apps")
                    try:
                        pyautogui.hotkey('win', 'd')  # Minimize all apps (Win + D)
                        last_action_time = current_time
                    except Exception as e:
                        logger.error(f"Failed to minimize apps: {e}")
                # Look for "Close" followed by "Open" in the buffer
                
                has_close=False
                for label in list(gesture_buffer)[:FRAME_WINDOW//2]:
                    if label =='Close':
                        logger.info(f"Frame {frame_count}: {predicted_label}, confidence={confidence:.2f}, {confidences}")
                        has_close=True
                        break
                has_open=False
                for label in list(gesture_buffer)[FRAME_WINDOW//2:]:
                    if label=='Open':
                        logger.info(f"Frame {frame_count}: {predicted_label}, confidence={confidence:.2f}, {confidences}")
                        has_open=True
                        break
                
                if has_close and has_open:
                    logger.debug(f"Gesture buffer content: {list(gesture_buffer)}")
                    logger.info("Detected Close -> Open transition, restoring apps")
                    try:
                        pyautogui.hotkey('win','d')
                        last_action_time=current_time
                    except Exception as e:
                        logger.error(f"Failed to restore apps: {e}")

            cv2.putText(frame, f"{predicted_label}: {confidence:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if confidence < 0.5 and confidence >= 0.4 and frame_count % 30 == 0:
                logger.info(f"Frame {frame_count}: {predicted_label}, confidence={confidence:.2f}, {confidences}")
                debug_path = f"debug_frames/frame_{frame_count}.jpg"
                cv2.imwrite(debug_path, frame)
                logger.info(f"Saved debug image: {debug_path}")

        else:
            gesture_buffer.append(None)  # No hand detected, add None to buffer
            cv2.putText(frame, "No hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        #displays the FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow('Hand Gesture Recognition', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    except Exception as e:
        logger.info(f"Frame {frame_count}: {predicted_label}, confidence={confidence:.2f}, {confidences}")
        logger.error(f"Error on frame {frame_count}: {e}")
        debug_path = f"debug_frames/error_frame_{frame_count}.jpg"
        cv2.imwrite(debug_path, frame)
        logger.info(f"Saved error debug image: {debug_path}")

cap.release()
cv2.destroyAllWindows()
hands.close()
logger.info("Cleaned up resources")