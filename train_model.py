"""
VITALSIGN — LSTM Training Script
Trains a single shared gloss-recognition model on ALL sequences found in
Data_landmarks/ — regardless of which teammate (signer) recorded them.

HOW THIS WORKS WITH YOUR TEAM WORKFLOW:
This script doesn't care who recorded what. It scans every sequence folder
under Data_landmarks/<action>/<any_signer>_<any_number>/ and treats them
all as training examples for that action. So once you `git pull` and have
everyone's Data_landmarks merged locally, just run this once — no code
changes needed regardless of how many people contributed data.

HOW TO USE:
1. Make sure Data_landmarks/ contains everyone's merged recordings.
2. Run: python train_model.py
3. Output: action.h5 (the trained model) and labels.json (action name <-> index mapping)
"""

import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# ============ CONFIG ============
DATA_PATH = "Data_landmarks"
SEQUENCE_LENGTH = 30
MODEL_OUTPUT_PATH = "action.keras"
LABELS_OUTPUT_PATH = "labels.json"
TEST_SPLIT = 0.15       # 15% held out for validation
RANDOM_SEED = 42
EPOCHS = 200             # early stopping will likely stop well before this
BATCH_SIZE = 16
# =================================


def discover_actions_and_sequences():
    """Scans Data_landmarks/ and returns {action_name: [list of sequence folder paths]}."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Couldn't find '{DATA_PATH}'. Run extract_landmarks.py first.")

    actions = sorted([d for d in os.listdir(DATA_PATH)
                       if os.path.isdir(os.path.join(DATA_PATH, d))])
    if not actions:
        raise ValueError(f"No action folders found inside '{DATA_PATH}'.")

    action_sequences = {}
    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        sequences = sorted([d for d in os.listdir(action_path)
                             if os.path.isdir(os.path.join(action_path, d))])
        action_sequences[action] = [os.path.join(action_path, s) for s in sequences]

    return actions, action_sequences


def load_sequence(seq_folder):
    """Loads all 30 frame .npy files from one sequence folder into a (30, feature_dim) array."""
    frames = []
    for frame_num in range(SEQUENCE_LENGTH):
        frame_path = os.path.join(seq_folder, f"{frame_num}.npy")
        if not os.path.exists(frame_path):
            raise FileNotFoundError(f"Missing frame file: {frame_path}")
        frames.append(np.load(frame_path))
    return np.array(frames)


def build_dataset():
    actions, action_sequences = discover_actions_and_sequences()
    label_map = {action: idx for idx, action in enumerate(actions)}

    print(f"Found {len(actions)} action(s): {actions}")
    for action in actions:
        count = len(action_sequences[action])
        print(f"  {action}: {count} sequence(s) (across all signers)")
        if count == 0:
            print(f"  ⚠  WARNING: '{action}' has zero usable sequences — it will be excluded.")

    X, y = [], []
    skipped = 0
    for action in actions:
        for seq_folder in action_sequences[action]:
            try:
                sequence = load_sequence(seq_folder)
                X.append(sequence)
                y.append(label_map[action])
            except FileNotFoundError as e:
                print(f"  Skipping incomplete sequence: {e}")
                skipped += 1

    if skipped:
        print(f"\nSkipped {skipped} incomplete sequence(s) (missing frame files).")

    if len(X) == 0:
        raise ValueError("No usable sequences found. Check your Data_landmarks folder.")

    X = np.array(X)
    y = to_categorical(np.array(y), num_classes=len(actions))

    print(f"\nFinal dataset shape: X={X.shape}, y={y.shape}")
    return X, y, actions, label_map


def build_model(sequence_length, feature_dim, num_classes):
    model = Sequential([
        Input(shape=(sequence_length, feature_dim)),
        LSTM(64, return_sequences=True, activation="tanh"),
        Dropout(0.2),
        LSTM(128, return_sequences=True, activation="tanh"),
        Dropout(0.2),
        LSTM(64, return_sequences=False, activation="tanh"),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
    return model


def main():
    X, y, actions, label_map = build_dataset()

    if len(X) < 20:
        print(f"\n⚠  WARNING: Only {len(X)} total sequences found across all actions.")
        print("   This is quite small for training. Results may not be reliable yet —")
        print("   consider this an early test run, not your final model.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\nTrain: {X_train.shape[0]} sequences | Test: {X_test.shape[0]} sequences")

    feature_dim = X.shape[2]
    model = build_model(SEQUENCE_LENGTH, feature_dim, len(actions))
    model.summary()

    early_stop = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1,
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal validation accuracy: {acc * 100:.1f}%")

    model.save(MODEL_OUTPUT_PATH)
    with open(LABELS_OUTPUT_PATH, "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\n✓ Model saved to '{MODEL_OUTPUT_PATH}'")
    print(f"✓ Label mapping saved to '{LABELS_OUTPUT_PATH}'")
    print("\nNext: use these two files for real-time inference in your web app.")


if __name__ == "__main__":
    main()