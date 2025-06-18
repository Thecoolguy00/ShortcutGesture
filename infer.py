import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import logging
import os
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('infer_log.txt', mode='w')
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

labels = ['Open', 'Close', 'Pointer','Victory']

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logger.error("Could not open webcam")
    exit()

frame_count = 0
os.makedirs('debug_frames', exist_ok=True)

while True:
    try:
        ret, frame = cap.read()
        frame_count += 1
        if not ret:
            logger.warning(f"Failed to read frame {frame_count}")
            continue

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        keypoints = []
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            raw_keypoints = []
            for i in range(21):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                raw_keypoints.extend([x, y])
            raw_keypoints = np.array(raw_keypoints, dtype=np.float32)
            logger.debug(f"Raw keypoints: min={raw_keypoints.min():.3f}, max={raw_keypoints.max():.3f}")

            keypoints = pre_process_landmark(raw_keypoints)
            keypoints = np.array(keypoints, dtype=np.float32)
            logger.debug(f"Normalized keypoints: min={keypoints.min():.3f}, max={keypoints.max():.3f}")

            keypoints_np = keypoints.reshape(1, 42)
            interpreter.set_tensor(input_details[0]['index'], keypoints_np)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_class = np.argmax(output_data[0])
            predicted_label = labels[predicted_class]
            confidence = output_data[0][predicted_class]
            confidences = {labels[i]: f"{output_data[0][i]:.2f}" for i in range(len(labels))}

            logger.info(f"Frame {frame_count}: {predicted_label}, confidence={confidence:.2f}, {confidences}")

            cv2.putText(frame, f"{predicted_label}: {confidence:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if confidence < 0.7:
                debug_path = f"debug_frames/frame_{frame_count}.jpg"
                cv2.imwrite(debug_path, frame)
                logger.info(f"Saved debug image: {debug_path}")

        else:
            cv2.putText(frame, "No hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Hand Gesture Recognition', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    except Exception as e:
        logger.error(f"Error on frame {frame_count}: {e}")
        debug_path = f"debug_frames/error_frame_{frame_count}.jpg"
        cv2.imwrite(debug_path, frame)
        logger.info(f"Saved error debug image: {debug_path}")

cap.release()
cv2.destroyAllWindows()
hands.close()
logger.info("Cleaned up resources")