"""
VITALSIGN — Landmark Verification Script
Quickly sanity-checks your extracted .npy landmark data before training.

USAGE:
python verify_landmarks.py
"""

import numpy as np
import os

LANDMARK_DATA_PATH = "Data_landmarks"
SEQUENCE_LENGTH = 30

# Expected vector length: pose(33*3) + face(468*3) + left_hand(21*3) + right_hand(21*3)
EXPECTED_LENGTH = 33 * 3 + 468 * 3 + 21 * 3 + 21 * 3  # = 1629

# Index ranges within the flattened vector, so we can check each part separately
POSE_RANGE = (0, 33 * 3)
FACE_RANGE = (POSE_RANGE[1], POSE_RANGE[1] + 468 * 3)
LEFT_HAND_RANGE = (FACE_RANGE[1], FACE_RANGE[1] + 21 * 3)
RIGHT_HAND_RANGE = (LEFT_HAND_RANGE[1], LEFT_HAND_RANGE[1] + 21 * 3)


def check_one_file(path):
    arr = np.load(path)
    issues = []

    if arr.shape[0] != EXPECTED_LENGTH:
        issues.append(f"Unexpected length {arr.shape[0]} (expected {EXPECTED_LENGTH})")

    pose = arr[POSE_RANGE[0]:POSE_RANGE[1]]
    face = arr[FACE_RANGE[0]:FACE_RANGE[1]]
    lh = arr[LEFT_HAND_RANGE[0]:LEFT_HAND_RANGE[1]]
    rh = arr[RIGHT_HAND_RANGE[0]:RIGHT_HAND_RANGE[1]]

    return {
        "path": path,
        "length": arr.shape[0],
        "issues": issues,
        "pose_all_zero": np.all(pose == 0),
        "face_all_zero": np.all(face == 0),
        "left_hand_all_zero": np.all(lh == 0),
        "right_hand_all_zero": np.all(rh == 0),
        "any_nan": np.any(np.isnan(arr)),
    }


def main():
    if not os.path.exists(LANDMARK_DATA_PATH):
        print(f"Couldn't find '{LANDMARK_DATA_PATH}'. Run extract_landmarks.py first.")
        return

    actions = sorted(os.listdir(LANDMARK_DATA_PATH))
    print(f"Found {len(actions)} action(s): {actions}\n")

    total_frames_checked = 0
    frames_with_no_hands = 0
    frames_with_no_pose = 0
    frames_with_no_face = 0
    frames_with_nan = 0
    length_mismatches = 0

    for action in actions:
        action_path = os.path.join(LANDMARK_DATA_PATH, action)
        if not os.path.isdir(action_path):
            continue

        sequences = sorted(os.listdir(action_path))
        print(f"[{action}] {len(sequences)} sequence(s) found")

        # Deep-check just the first sequence of each action to keep this fast
        sample_seq = sequences[0]
        sample_path = os.path.join(action_path, sample_seq)
        print(f"  Sampling sequence '{sample_seq}':")

        for frame_num in range(SEQUENCE_LENGTH):
            frame_path = os.path.join(sample_path, f"{frame_num}.npy")
            if not os.path.exists(frame_path):
                print(f"    Frame {frame_num}: MISSING FILE")
                continue

            result = check_one_file(frame_path)
            total_frames_checked += 1

            if result["issues"]:
                length_mismatches += 1
                print(f"    Frame {frame_num}: {result['issues']}")

            if result["left_hand_all_zero"] and result["right_hand_all_zero"]:
                frames_with_no_hands += 1

            if result["pose_all_zero"]:
                frames_with_no_pose += 1

            if result["face_all_zero"]:
                frames_with_no_face += 1

            if result["any_nan"]:
                frames_with_nan += 1

        # Quick one-line summary for this action's sampled sequence
        print(f"    -> checked {SEQUENCE_LENGTH} frames from this sequence")

    print("\n" + "=" * 50)
    print("SUMMARY")
    print(f"Total frames checked: {total_frames_checked}")
    print(f"Frames with NO hand detected at all (both hands zero): {frames_with_no_hands}")
    print(f"Frames with NO pose detected: {frames_with_no_pose}")
    print(f"Frames with NO face detected: {frames_with_no_face}")
    print(f"Frames with NaN values: {frames_with_nan}")
    print(f"Frames with wrong vector length: {length_mismatches}")

    if frames_with_no_hands > total_frames_checked * 0.3:
        print("\n⚠  WARNING: Over 30% of sampled frames have no hand landmarks detected.")
        print("   This usually means hands were out of frame, too fast, or poorly lit")
        print("   during recording. Consider re-recording affected words.")
    elif frames_with_nan > 0:
        print("\n⚠  WARNING: NaN values found — something went wrong during extraction.")
    elif length_mismatches > 0:
        print("\n⚠  WARNING: Some frames have unexpected vector lengths — check extract_landmarks.py.")
    else:
        print("\n✓ Looks healthy! Your landmark data appears ready for training.")


if __name__ == "__main__":
    main()