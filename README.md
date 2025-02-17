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



## Required Files & Dependencies
### **1. Files & Directory Structure**
```
/driver-monitoring
│── app.py                # Main application script
│── requirements.txt       # List of dependencies
│── shape_predictor_68_face_landmarks.dat  # Dlib model
│── templates/
│   └── index.html         # Web interface template
│── static/
│   └── style.css          # Stylesheet (optional)
│── config.py              # Configuration parameters
│── README.md              # Project Documentation
```

### **2. Required Dependencies**
Install dependencies using:
```bash
pip install -r requirements.txt
```
Or install manually:
```bash
pip install opencv-python dlib imutils flask numpy scipy
```

### **3. Download Dlib Model**
Download the **68-face landmark model**:
```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```
Place this file in the project directory.

---

## How to Run the Project
### **1. Clone the Repository**
```bash
git clone https://github.com/your-repo/driver-monitoring.git
cd driver-monitoring
```

### **2. Run the Application**
```bash
python app.py
```

### **3. Access the Web Interface**
Open your browser and go to:
```
http://127.0.0.1:5000/
```

---

## Configuration
Modify **config.py** to adjust thresholds:
```python
class Config:
    EYE_AR_THRESH = 0.27  # Eye Aspect Ratio Threshold
    EYE_AR_CONSEC_FRAMES = 17  # Consecutive Frames for Drowsiness
    YAWN_THRESH = 20  # Lip Distance Threshold for Yawning
    YAWN_CONSEC_FRAMES = 7  # Consecutive Frames for Yawning
```


## Troubleshooting
- **No Video Feed?** Ensure your webcam is connected.
- **Dlib Model Not Found?** Download and place `shape_predictor_68_face_landmarks.dat` in the project folder.
- **Errors in Flask?** Ensure Flask is installed and running on Python 3.8+.


## Future Improvements
- **Sound Alerts** for drowsiness detection.
- **Mobile App Integration** for remote monitoring.
- **Better Performance Optimization** for real-time processing.


## Acknowledgments
- OpenCV for real-time video processing.
- Dlib for facial landmark detection.
- SciPy for Euclidean distance calculation.

