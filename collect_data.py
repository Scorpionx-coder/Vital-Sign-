"""
VITALSIGN — Data Collection Script
Records webcam sequences, extracts MediaPipe Holistic landmarks,
and saves them as labeled .npy files for LSTM training.

HOW TO USE:
1. Set SIGNER_ID below to your own initials (e.g. "AS", "AD", "GS")
2. Set ACTIONS to the words you're recording right now (start with 2-3 to test)
3. Run: python collect_data.py
4. Follow on-screen prompts. Press 'q' at any time to quit early.
"""

import cv2
import numpy as np
import os
import mediapipe as mp

# ============ CONFIG — EDIT THESE BEFORE RUNNING ============
SIGNER_ID = "AS"          # <-- CHANGE THIS to your initials before running
ACTIONS = ["hello", "pain", "help"]   # <-- start small to test, expand to all 50 later
NO_SEQUENCES = 15          # sequences per word (increase later once tested)
SEQUENCE_LENGTH = 30       # frames per sequence
DATA_PATH = "Data"         # root folder where everything gets saved
# ==============================================================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


def extract_keypoints(results):
    pose = np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[r.x, r.y, r.z] for r in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])


def setup_folders():
    for action in ACTIONS:
        for seq in range(NO_SEQUENCES):
            folder = os.path.join(DATA_PATH, action, f"{SIGNER_ID}_{seq}")
            os.makedirs(folder, exist_ok=True)


def main():
    setup_folders()
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in ACTIONS:
            for sequence in range(NO_SEQUENCES):
                for frame_num in range(SEQUENCE_LENGTH):

                    ret, frame = cap.read()
                    if not ret:
                        print("Webcam not detected — check your camera connection.")
                        cap.release()
                        return

                    image, results = mediapipe_detection(frame, holistic)
                    draw_landmarks(image, results)

                    # On-screen guidance
                    if frame_num == 0:
                        cv2.putText(image, f'Get ready: "{action}"  (seq {sequence})', (15, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.imshow("VITALSIGN Data Collection", image)
                        cv2.waitKey(1500)  # 1.5s pause to get into position
                    else:
                        cv2.putText(image, f'Recording "{action}"  seq {sequence}  frame {frame_num}',
                                    (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.imshow("VITALSIGN Data Collection", image)

                    keypoints = extract_keypoints(results)
                    save_path = os.path.join(DATA_PATH, action, f"{SIGNER_ID}_{sequence}", f"{frame_num}.npy")
                    np.save(save_path, keypoints)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

    cap.release()
    cv2.destroyAllWindows()
    print("Done! Check the 'Data' folder for your recorded sequences.")


if __name__ == "__main__":
    main()