import cv2
import mediapipe as mp
import pyautogui
import time

class EyeController:
    def __init__(self):
        # Initialize MediaPipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)

        # Landmark indices
        self.LEFT_IRIS = [468]
        self.RIGHT_IRIS = [473]
        self.LEFT_EYE = [159, 145]
        self.RIGHT_EYE = [386, 374]
        self.LEFT_EYE_LR = [33, 133]
        self.RIGHT_EYE_LR = [362, 263]

        # Blink detection thresholds and timers
        self.blink_ratio_threshold = 0.25
        self.left_blink_time = 0
        self.right_blink_time = 0

        # Scroll control
        self.prev_head_y = None
        self.scroll_cooldown = 1.0  # seconds
        self.last_scroll_time = 0

        # Screen size
        self.screen_w, self.screen_h = pyautogui.size()

    def get_eye_center(self, landmarks, indices, frame_w, frame_h):
        x = int(landmarks[indices[0]].x * frame_w)
        y = int(landmarks[indices[0]].y * frame_h)
        return x, y

    def get_eye_ratio(self, landmarks, top, bottom, left, right, h, w):
        vertical = abs(landmarks[top].y - landmarks[bottom].y) * h
        horizontal = abs(landmarks[left].x - landmarks[right].x) * w
        return vertical / horizontal if horizontal != 0 else 0

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        h, w, _ = frame.shape

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Eye centers
            left_iris = self.get_eye_center(landmarks, self.LEFT_IRIS, w, h)
            right_iris = self.get_eye_center(landmarks, self.RIGHT_IRIS, w, h)

            # Cursor movement
            cx = (left_iris[0] + right_iris[0]) // 2
            cy = (left_iris[1] + right_iris[1]) // 2

            scale = 3.0  # sensitivity
            dx = (cx - w // 2) * scale
            dy = (cy - h // 2) * scale

            screen_x = int(self.screen_w // 2 + dx)
            screen_y = int(self.screen_h // 2 + dy)
            pyautogui.moveTo(screen_x, screen_y, duration=0.1)

            # Eye blink detection
            left_ratio = self.get_eye_ratio(landmarks, 159, 145, 33, 133, h, w)
            right_ratio = self.get_eye_ratio(landmarks, 386, 374, 362, 263, h, w)

            chin_y = landmarks[152].y * h
            now = time.time()

            # Left eye blink = Left click
            if left_ratio < self.blink_ratio_threshold and (now - self.left_blink_time > 1):
                pyautogui.click(button='left')
                self.left_blink_time = now
                print("Left click")

            # Right eye blink = Right click
            if right_ratio < self.blink_ratio_threshold and (now - self.right_blink_time > 1):
                pyautogui.click(button='right')
                self.right_blink_time = now
                print("Right click")

            # Scroll with both eyes closed and head movement
            if left_ratio < self.blink_ratio_threshold and right_ratio < self.blink_ratio_threshold:
                if self.prev_head_y is not None and (now - self.last_scroll_time) > self.scroll_cooldown:
                    dy = chin_y - self.prev_head_y
                    if dy < -5:
                        pyautogui.scroll(30)
                        print("⬆ Scroll Up")
                        self.last_scroll_time = now
                    elif dy > 5:
                        pyautogui.scroll(-30)
                        print("⬇ Scroll Down")
                        self.last_scroll_time = now

            self.prev_head_y = chin_y

            # Visual feedback
            cv2.circle(frame, left_iris, 5, (0, 255, 0), -1)
            cv2.circle(frame, right_iris, 5, (0, 255, 0), -1)
            cv2.putText(frame, f"Left EAR: {left_ratio:.2f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"Right EAR: {right_ratio:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"Head Y: {int(chin_y)}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame
