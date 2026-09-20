"""
VITALSIGN — Data Collection Script (v3 — with on-screen framing guide)

WHAT'S NEW IN THIS VERSION:
A visual framing guide now shows on screen at all times, so you can see
whether you're correctly positioned BEFORE and WHILE recording. This fixes
a real issue we hit: MediaPipe Holistic detects your pose (shoulders/torso)
FIRST, then uses that to locate your face and hands. If your shoulders
aren't visible in frame, pose detection fails — and face/hand detection
fail right along with it, even if your face is clearly visible to a human.

HOW TO USE:
1. Run: python collect_data.py
2. First time only: type your initials when asked (e.g. AS, AD, GS).
3. Sit back so your head AND shoulders fit inside the green guide box.
   A red box means you're too close or not centered — reposition before
   recording starts.
4. Follow on-screen prompts. Press 'q' at any time to quit early.

NOTE: This version saves raw frames. Extract landmarks later using
extract_landmarks.py
"""

import cv2
import numpy as np
import os
import re

# ============ CONFIG — EDIT THESE BEFORE RUNNING ============
ACTIONS = ["hello", "pain", "help"]   # <-- start small to test, expand to all 50 later
SEQUENCES_TO_RECORD_NOW = 15          # how many NEW sequences to add this run
SEQUENCE_LENGTH = 30                  # frames per sequence
DATA_PATH = "Data"                    # root folder where everything gets saved
SIGNER_ID_FILE = ".signer_id"         # local file that remembers your ID (auto-created)
# ==============================================================


def get_signer_id():
    """Reads the signer ID from a local file, or asks once and saves it."""
    if os.path.exists(SIGNER_ID_FILE):
        with open(SIGNER_ID_FILE, "r") as f:
            saved_id = f.read().strip()
            if saved_id:
                print(f"Using saved Signer ID: {saved_id}")
                return saved_id

    while True:
        signer_id = input("First time setup — enter your initials (2-4 letters, e.g. AS): ").strip().upper()
        if re.fullmatch(r"[A-Z]{2,4}", signer_id):
            with open(SIGNER_ID_FILE, "w") as f:
                f.write(signer_id)
            print(f"Saved! Your Signer ID '{signer_id}' will auto-load every time from now on.")
            return signer_id
        print("Please enter 2-4 letters only (no numbers or symbols).")


def get_next_sequence_number(action, signer_id):
    """Scans existing folders for this action+signer and returns the next free number."""
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        return 0
    existing = [d for d in os.listdir(action_path) if d.startswith(f"{signer_id}_")]
    if not existing:
        return 0
    numbers = [int(d.split("_")[-1]) for d in existing if d.split("_")[-1].isdigit()]
    return max(numbers) + 1 if numbers else 0


def draw_framing_guide(frame):
    """Draws a guide box showing where the head+shoulders should sit.
    Also draws a lower 'shoulder line' — if your shoulders are above this
    line (higher up in frame / too close to camera), you're too close."""
    h, w = frame.shape[:2]

    # Guide box: roughly centered, sized to encourage sitting back
    box_w, box_h = int(w * 0.55), int(h * 0.85)
    x1 = (w - box_w) // 2
    y1 = int(h * 0.05)
    x2 = x1 + box_w
    y2 = y1 + box_h

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Shoulder line: shoulders should be at or below this line
    shoulder_line_y = int(h * 0.55)
    cv2.line(frame, (x1, shoulder_line_y), (x2, shoulder_line_y), (0, 255, 255), 2)

    cv2.putText(frame, "Fit head+shoulders in green box", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Shoulders should be below this line", (x1, shoulder_line_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    return frame


def main():
    signer_id = get_signer_id()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Webcam not detected — check your camera connection.")
        return

    print("\n" + "=" * 60)
    print("FRAMING CHECK: Position yourself so your head AND shoulders")
    print("fit inside the green box, with shoulders below the yellow line.")
    print("Press 's' when you're ready to start recording, or 'q' to quit.")
    print("=" * 60 + "\n")

    # Framing preview loop — waits until the user confirms they're positioned well
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Webcam not detected — check your camera connection.")
            cap.release()
            return

        frame = draw_framing_guide(frame)
        cv2.putText(frame, "Press 's' to start recording, 'q' to quit", (15, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("VITALSIGN Data Collection", frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('s'):
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            print("Cancelled before recording started.")
            return

    try:
        for action in ACTIONS:
            start_seq = get_next_sequence_number(action, signer_id)
            print(f'"{action}": found existing sequences, continuing from #{start_seq}')

            for i in range(SEQUENCES_TO_RECORD_NOW):
                sequence = start_seq + i
                folder = os.path.join(DATA_PATH, action, f"{signer_id}_{sequence}")
                os.makedirs(folder, exist_ok=True)

                for frame_num in range(SEQUENCE_LENGTH):
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to read frame from webcam.")
                        cap.release()
                        return

                    # Keep an unmodified copy to save — the guide overlay below
                    # is only for the preview window, and must NOT be baked into
                    # the saved image (it would interfere with landmark detection).
                    clean_frame = frame.copy()
                    display_frame = draw_framing_guide(frame)

                    if frame_num == 0:
                        cv2.putText(display_frame, f'Get ready: "{action}"  (seq {sequence})', (15, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.imshow("VITALSIGN Data Collection", display_frame)
                        cv2.waitKey(1500)  # 1.5s pause to get into position
                    else:
                        cv2.putText(display_frame, f'Recording "{action}"  seq {sequence}  frame {frame_num}',
                                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.imshow("VITALSIGN Data Collection", display_frame)

                    # Save the CLEAN frame (no overlay) as JPG
                    save_path = os.path.join(folder, f"{frame_num:03d}.jpg")
                    cv2.imwrite(save_path, clean_frame)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        print("Recording stopped by user.")
                        return

        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Recording complete!")
        print("Check the 'Data' folder for your recorded sequences.")
        print("\nNext: Run 'python extract_landmarks.py' to extract landmarks from frames.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()