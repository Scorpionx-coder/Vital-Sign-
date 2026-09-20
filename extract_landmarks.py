"""
VITALSIGN — Landmark Extraction Script (v3 — new Tasks API)

WHY THIS CHANGED:
MediaPipe removed the classic `mediapipe.solutions.holistic` API in versions
0.10.31 and later. Since only 0.10.30+ is installable on current setups,
we now use the officially supported replacement: HolisticLandmarker from
MediaPipe's newer "Tasks" API. This is Google's actual recommended path
forward, not a workaround.

ONE-TIME SETUP BEFORE RUNNING:
1. Install dependencies:
   pip install mediapipe opencv-python numpy

2. Download the HolisticLandmarker model file and place it in this same
   project folder, named exactly "holistic_landmarker.task":

   PowerShell:
     Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task" -OutFile "holistic_landmarker.task"

   Or just paste that URL into your browser and save the file with that exact name.

USAGE:
1. Make sure collect_data.py has already saved frames into the 'Data' folder.
2. Run: python extract_landmarks.py
3. Output goes into 'Data_landmarks/', mirroring 'Data/' but with .npy files.

Safe to re-run — automatically skips sequences already fully processed.
"""

import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision

# ============ CONFIG ============
RAW_DATA_PATH = "Data"
LANDMARK_DATA_PATH = "Data_landmarks"
SEQUENCE_LENGTH = 30
MODEL_PATH = "holistic_landmarker.task"

POSE_LANDMARKS_COUNT = 33
FACE_LANDMARKS_COUNT = 468
HAND_LANDMARKS_COUNT = 21
# =================================


def build_landmarker():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Couldn't find '{MODEL_PATH}'. Download it first — see the "
            f"setup instructions at the top of this script."
        )
    return mp_vision.HolisticLandmarker.create_from_model_path(MODEL_PATH)


def landmarks_to_array(landmarks, count):
    """Converts a list of NormalizedLandmark objects into a flat x,y,z array.
    Returns zeros if nothing was detected (e.g. hand out of frame)."""
    if not landmarks:
        return np.zeros(count * 3)
    arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
    expected = count * 3
    if arr.shape[0] < expected:
        arr = np.concatenate([arr, np.zeros(expected - arr.shape[0])])
    elif arr.shape[0] > expected:
        arr = arr[:expected]
    return arr


def extract_keypoints(result):
    pose = landmarks_to_array(result.pose_landmarks, POSE_LANDMARKS_COUNT)
    face = landmarks_to_array(result.face_landmarks, FACE_LANDMARKS_COUNT)
    lh = landmarks_to_array(result.left_hand_landmarks, HAND_LANDMARKS_COUNT)
    rh = landmarks_to_array(result.right_hand_landmarks, HAND_LANDMARKS_COUNT)
    return np.concatenate([pose, face, lh, rh])


def sequence_already_done(output_folder):
    if not os.path.exists(output_folder):
        return False
    expected = {f"{i}.npy" for i in range(SEQUENCE_LENGTH)}
    existing = set(os.listdir(output_folder))
    return expected.issubset(existing)


def main():
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Couldn't find '{RAW_DATA_PATH}' folder. Run collect_data.py first.")
        return

    landmarker = build_landmarker()

    actions = sorted(os.listdir(RAW_DATA_PATH))
    total_sequences = 0
    processed_sequences = 0
    skipped_sequences = 0

    for action in actions:
        action_path = os.path.join(RAW_DATA_PATH, action)
        if not os.path.isdir(action_path):
            continue

        sequences = sorted(os.listdir(action_path))
        for seq_folder in sequences:
            seq_path = os.path.join(action_path, seq_folder)
            if not os.path.isdir(seq_path):
                continue

            total_sequences += 1
            output_folder = os.path.join(LANDMARK_DATA_PATH, action, seq_folder)

            if sequence_already_done(output_folder):
                skipped_sequences += 1
                continue

            os.makedirs(output_folder, exist_ok=True)
            print(f"Processing {action}/{seq_folder} ...")

            for frame_num in range(SEQUENCE_LENGTH):
                jpg_path = os.path.join(seq_path, f"{frame_num:03d}.jpg")

                if not os.path.exists(jpg_path):
                    print(f"  Warning: missing frame {jpg_path}, filling with zeros.")
                    keypoints = np.zeros(
                        POSE_LANDMARKS_COUNT * 3 + FACE_LANDMARKS_COUNT * 3
                        + HAND_LANDMARKS_COUNT * 3 + HAND_LANDMARKS_COUNT * 3
                    )
                else:
                    frame = cv2.imread(jpg_path)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    result = landmarker.detect(mp_image)
                    keypoints = extract_keypoints(result)

                np.save(os.path.join(output_folder, f"{frame_num}.npy"), keypoints)

            processed_sequences += 1

    landmarker.close()

    print("\n" + "=" * 50)
    print(f"Done! {processed_sequences} sequence(s) processed, "
          f"{skipped_sequences} already done (skipped), "
          f"{total_sequences} total found.")
    print(f"Landmark data saved in: '{LANDMARK_DATA_PATH}/'")
    print("Next: use this folder as input for your LSTM training script.")


if __name__ == "__main__":
    main()