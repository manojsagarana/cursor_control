from flask import Flask, render_template, Response, request
import cv2
from gesture_control import GestureController
from eye_cursor_control import EyeController

app = Flask(__name__)

# Initialize both controllers
gesture_controller = GestureController()
eye_controller = EyeController()

# Capture from the same camera (shared)
cap = cv2.VideoCapture(0)

# Mode selector (default is hand)
mode = "hand"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        global mode
        while True:
            success, frame = cap.read()
            if not success:
                break
            if mode == "hand":
                frame = gesture_controller.process_frame(frame)
            else:
                frame = eye_controller.process_frame(frame)
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_mode/<control_mode>')
def set_mode(control_mode):
    global mode
    if control_mode in ["hand", "eye"]:
        mode = control_mode
    return ("", 204)

if __name__ == '__main__':
    app.run(debug=True)