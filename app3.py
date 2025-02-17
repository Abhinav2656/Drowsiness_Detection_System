import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from imutils.video import VideoStream
from imutils import face_utils
from dataclasses import dataclass
from flask import Flask, Response, render_template_string

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Driver Monitoring System</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f2f5;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 20px;
            color: #1a237e;
        }
        .video-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            width: 100%;
            max-width: 640px;
            margin: 0 auto;
        }
        .video-feed {
            width: 100%;
            max-width: 640px;
            height: auto;
            border-radius: 10px;
            display: block;
        }
        .alert {
            background-color: #ff5252;
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin-top: 10px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            display: none;
            animation: blink 1s infinite;
        }
        @keyframes blink {
            50% { opacity: 0.5; }
        }
        #drowsinessAlert {
            background-color: #ff5252;
        }
        #yawnAlert {
            background-color: #ff9800;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Driver Monitoring System</h1>
        </div>
        <div class="video-container">
            <img src="{{ url_for('video_feed') }}" class="video-feed">
            <div id="drowsinessAlert" class="alert">
                ⚠️ DROWSINESS DETECTED - TAKE A BREAK! ⚠️
            </div>
            <div id="yawnAlert" class="alert">
                😴 FREQUENT YAWNING DETECTED - FEELING TIRED? 😴
            </div>
        </div>
    </div>

    <script>
        function checkAlerts() {
            fetch('/check_alerts')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('drowsinessAlert').style.display = 
                        data.is_drowsy ? 'block' : 'none';
                    document.getElementById('yawnAlert').style.display = 
                        data.is_yawning ? 'block' : 'none';
                });
        }
        setInterval(checkAlerts, 300);  // Check more frequently
    </script>
</body>
</html>
'''


@dataclass
class Config:
    """Configuration parameters for detection with more sensitive thresholds"""
    EYE_AR_THRESH: float = 0.27 # Increased threshold for easier detection
    EYE_AR_CONSEC_FRAMES: int = 17  # Reduced frames for quicker detection
    YAWN_THRESH: float = 20  # Reduced threshold for easier yawn detection
    YAWN_CONSEC_FRAMES: int = 7  # Reduced frames for quicker yawn detection
    DEFAULT_ALARM_PATH: str = "Alert.WAV"


class FacialFeatures:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    def calculate_eye_aspect_ratio(self, eye_points):
        # Vertical distances
        v1 = distance.euclidean(eye_points[1], eye_points[5])
        v2 = distance.euclidean(eye_points[2], eye_points[4])
        # Horizontal distance
        h = distance.euclidean(eye_points[0], eye_points[3])
        # Add small epsilon to prevent division by zero
        return (v1 + v2) / (2.0 * h + 1e-6)

    def get_eye_measurements(self, shape):
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = self.calculate_eye_aspect_ratio(leftEye)
        rightEAR = self.calculate_eye_aspect_ratio(rightEye)

        # Use minimum EAR for more sensitive detection
        return (min(leftEAR, rightEAR), leftEye, rightEye)

    def calculate_lip_distance(self, shape):
        # Enhanced lip distance calculation
        top_lip = shape[50:53]
        top_lip = np.concatenate((top_lip, shape[61:64]))
        bottom_lip = shape[56:59]
        bottom_lip = np.concatenate((bottom_lip, shape[65:68]))

        top_mean = np.mean(top_lip, axis=0)
        bottom_mean = np.mean(bottom_lip, axis=0)

        return abs(top_mean[1] - bottom_mean[1])


class DrowsinessDetector:
    def __init__(self):
        self.config = Config()
        self.facial_features = FacialFeatures()
        self.eye_counter = 0
        self.yawn_counter = 0
        self.is_drowsy = False
        self.is_yawning = False
        self.ear_history = []
        self.lip_history = []

    def process_frame(self, frame):
        if frame is None:
            return frame

        # Resize frame for faster processing while maintaining quality
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Enhance contrast
        gray = cv2.equalizeHist(gray)

        # Detect faces with different scale parameters
        faces = self.facial_features.detector(gray, 0)

        for face in faces:
            shape = self.facial_features.predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            # Get eye measurements
            ear, leftEye, rightEye = self.facial_features.get_eye_measurements(shape)
            self.ear_history.append(ear)
            if len(self.ear_history) > 5:  # Keep last 5 frames
                self.ear_history.pop(0)

            # Get lip distance
            lip_distance = self.facial_features.calculate_lip_distance(shape)
            self.lip_history.append(lip_distance)
            if len(self.lip_history) > 5:
                self.lip_history.pop(0)

            # Use smoothed values for more stable detection
            avg_ear = np.mean(self.ear_history)
            avg_lip_distance = np.mean(self.lip_history)

            # Draw facial landmarks
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)

            # Color-coded contours based on state
            eye_color = (0, 0, 255) if avg_ear < self.config.EYE_AR_THRESH else (0, 255, 0)
            lip_color = (0, 0, 255) if avg_lip_distance > self.config.YAWN_THRESH else (0, 255, 0)

            cv2.drawContours(frame, [leftEyeHull], -1, eye_color, 2)
            cv2.drawContours(frame, [rightEyeHull], -1, eye_color, 2)
            cv2.drawContours(frame, [shape[48:60]], -1, lip_color, 2)

            # Check for drowsiness with smoothed EAR
            if avg_ear < self.config.EYE_AR_THRESH:
                self.eye_counter += 2  # Increment faster
                if self.eye_counter >= self.config.EYE_AR_CONSEC_FRAMES:
                    self.is_drowsy = True
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.eye_counter = max(0, self.eye_counter - 1)  # Gradual decrease
                if self.eye_counter < self.config.EYE_AR_CONSEC_FRAMES:
                    self.is_drowsy = False

            # Check for yawning with smoothed lip distance
            if avg_lip_distance > self.config.YAWN_THRESH:
                self.yawn_counter += 2  # Increment faster
                if self.yawn_counter >= self.config.YAWN_CONSEC_FRAMES:
                    self.is_yawning = True
                    cv2.putText(frame, "YAWN ALERT!", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.yawn_counter = max(0, self.yawn_counter - 1)  # Gradual decrease
                if self.yawn_counter < self.config.YAWN_CONSEC_FRAMES:
                    self.is_yawning = False

            # Display metrics with color coding
            ear_color = (0, 0, 255) if avg_ear < self.config.EYE_AR_THRESH else (255, 255, 255)
            lip_color = (0, 0, 255) if avg_lip_distance > self.config.YAWN_THRESH else (255, 255, 255)

            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, ear_color, 2)
            cv2.putText(frame, f"LIP: {avg_lip_distance:.2f}", (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, lip_color, 2)

        return frame


detector = DrowsinessDetector()
video_stream = VideoStream(src=0).start()


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


def generate_frames():
    while True:
        frame = video_stream.read()
        if frame is None:
            break
        frame = detector.process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/check_alerts')
def check_alerts():
    return {
        "is_drowsy": detector.is_drowsy,
        "is_yawning": detector.is_yawning
    }


if __name__ == '__main__':
    try:
        app.run(debug=False, port=5000)
    finally:
        video_stream.stop()