# Vital-Sign-# VITALSIGN

Domain-Specific Indian Sign Language Recognition for Emergency Medical Triage, using MediaPipe landmark extraction and an LSTM-based gloss recognition model.

## Team
**The Decodders** — Amandeep Singh, Abhijeet Dhiman, Gurpreet Singh
Supervised by Dr. Neeraj Mohan

---

## ⚠️ Read This Before You Start (Saves You Hours)

- **Use Python 3.11.** MediaPipe does not work correctly on Python 3.13 — you'll get an `AttributeError: module 'mediapipe' has no attribute 'solutions'` type of error. Even on a supported version, current mediapipe releases (0.10.31+) removed the old `solutions` API entirely — our scripts already use the newer replacement, so this only matters if you're editing code, not running it.
- **Frame yourself correctly when recording.** Sit back so your **shoulders and upper chest are visible**, not just your face in close-up. `collect_data.py` shows an on-screen green box + yellow line as a guide — follow it. If you skip this, landmark detection will fail on most frames (we lost a full test session to this early on).
- **Get signs from the real ISL dictionary**, not from guessing: [indiansignlanguage.org](https://indiansignlanguage.org/). Don't record a word until you've watched the actual demonstration video.

---

## One-Time Setup (Do This First)

### 1. Install Python 3.11
Download from [python.org/downloads](https://python.org/downloads) — during install, check **"Add python.exe to PATH"**. You can keep other Python versions installed too; they won't conflict.

### 2. Clone the repo
```bash
git clone <repo-url>
cd Vital-Sign-
```

### 3. Create and activate a virtual environment (using Python 3.11 specifically)
```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
You'll know it worked when your terminal prompt starts with `(venv)`.

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Download the MediaPipe Holistic model file
Required for `extract_landmarks.py`. Run this from inside the project folder:
```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task" -OutFile "holistic_landmarker.task"
```

### 6. Verify everything works
```bash
python -c "import mediapipe as mp; print(mp.tasks.python.vision.HolisticLandmarker)"
```
If this prints something instead of erroring, you're ready.

---

## The Full Pipeline (Run in This Order)

| Step | Script | What it does |
|---|---|---|
| 1 | `collect_data.py` | Records webcam sequences into `Data/`, one folder per word per sequence |
| 2 | `extract_landmarks.py` | Reads those frames, runs MediaPipe, saves landmark arrays into `Data_landmarks/` |
| 3 | `verify_landmarks.py` | Sanity-checks the extracted data before training (detection rates, no NaNs, etc.) |
| 4 | `train_model.py` | Trains the LSTM model on everything in `Data_landmarks/`, saves `action.keras` + `labels.json` |

### Recording your part
```bash
python collect_data.py
```
- First run asks for your initials (e.g. `AS`, `AD`, `GS`) — this tags all your recordings so multiple people's data can merge without overwriting each other.
- A framing guide box appears — position yourself correctly, press `s` to begin.
- Edit the `ACTIONS` list at the top of the script to whichever words you're recording.

### After recording, extract and verify
```bash
python extract_landmarks.py
python verify_landmarks.py
```
Check the verify output — if a high percentage of frames show "no hand/pose/face detected," re-check your framing before recording more.

---

## How We're Merging Data Across 3 Laptops

We are **not** training 3 separate models. Each person records + extracts locally, then:
- **`Data_landmarks/` is shared via Google Drive**, not pushed through Git (large binary files caused repeated push timeouts — learned that one the hard way).
- Zip your `Data_landmarks` folder and upload it to the shared team Drive folder.
- Whoever does the final training run downloads everyone's zips and merges the folders together locally (no conflicts, since folder names are signer-prefixed).
- `train_model.py` automatically picks up everyone's sequences per word — no code changes needed regardless of who contributed what.

---

## Known Gotchas We Already Hit (So You Don't Have To)

- **Git push timing out on large files (HTTP 408):** don't push `Data/` or `Data_landmarks/` through Git — they're already in `.gitignore`. Share via Drive instead.
- **Mislabeled recordings:** if you record the wrong sign for a word, don't just edit `labels.json` — that gets overwritten every time `train_model.py` runs. Rename the actual folder to match what was really performed.
- **"No attribute solutions" mediapipe error:** means either Python 3.13, or an unpinned `mediapipe` install pulling a version too new for the old API. Our scripts already use the newer Tasks API to sidestep this entirely — just make sure your `requirements.txt` install went through cleanly.