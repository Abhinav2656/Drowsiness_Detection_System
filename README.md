# Driver Monitoring System

## Overview
The Driver Monitoring System is a real-time AI-based application that detects driver drowsiness and yawning using computer vision techniques. It leverages OpenCV, Dlib, and Flask to process video feeds and provide visual and auditory alerts when signs of fatigue are detected.

## Features
- **Eye Aspect Ratio (EAR) Calculation**: Detects drowsiness by monitoring the eye-blinking rate.
- **Lip Distance Measurement**: Identifies yawning based on mouth opening.
- **Live Video Streaming**: Integrates with Flask for real-time web-based monitoring.
- **Alert System**: Displays visual alerts when drowsiness or yawning is detected.

## Installation
### Prerequisites
Ensure you have the following dependencies installed:
- Python 3.8+
- OpenCV
- Dlib
- imutils
- Flask
- NumPy
- SciPy

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/driver-monitoring.git
   cd driver-monitoring
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the required dlib facial landmark model:
   ```bash
   wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
   ```

## Usage
1. Start the Flask server:
   ```bash
   python app.py
   ```
2. Open a browser and visit:
   ```
   http://127.0.0.1:5000/
   ```
3. The web interface will display the live video feed and issue alerts if drowsiness or yawning is detected.

## Configuration
Modify thresholds in `config.py`:
- `EYE_AR_THRESH` (default: `0.27`): Eye aspect ratio threshold for drowsiness detection.
- `EYE_AR_CONSEC_FRAMES` (default: `17`): Number of frames before drowsiness alert.
- `YAWN_THRESH` (default: `20`): Lip distance threshold for yawning detection.
- `YAWN_CONSEC_FRAMES` (default: `7`): Number of frames before yawning alert.

## Acknowledgments
- OpenCV for real-time video processing.
- Dlib for facial landmark detection.
- SciPy for Euclidean distance calculation.

