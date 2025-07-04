import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

class GestureController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.75)
        self.mp_draw = mp.solutions.drawing_utils
        self.screen_w, self.screen_h = pyautogui.size()
        self.gesture_cooldown = 0
        self.scroll_cooldown = 0
        self.tab_cooldown = 0
        self.cursor_history = []

    def fingers_up(self, lm):
        return [
            lm[4].x < lm[3].x,
            lm[8].y < lm[6].y,
            lm[12].y < lm[10].y,
            lm[16].y < lm[14].y,
            lm[20].y < lm[18].y
        ]

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)

        self.gesture_cooldown = max(0, self.gesture_cooldown - 1)
        self.scroll_cooldown = max(0, self.scroll_cooldown - 1)
        self.tab_cooldown = max(0, self.tab_cooldown - 1)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                lm = hand_landmarks.landmark
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                fingers = self.fingers_up(lm)

                if fingers == [True, True, True, True, True]:
                    x, y = int(lm[8].x * w), int(lm[8].y * h)
                    margin = 100
                    screen_x = np.interp(x, [margin, w - margin], [0, self.screen_w])
                    screen_y = np.interp(y, [margin, h - margin], [0, self.screen_h])

                    self.cursor_history.append((screen_x, screen_y))
                    if len(self.cursor_history) > 5:
                        self.cursor_history.pop(0)

                    avg_x = sum(pt[0] for pt in self.cursor_history) / len(self.cursor_history)
                    avg_y = sum(pt[1] for pt in self.cursor_history) / len(self.cursor_history)
                    pyautogui.moveTo(avg_x, avg_y, duration=0.002)

                if fingers == [False, True, False, False, False] and self.gesture_cooldown == 0:
                    pyautogui.click()
                    cv2.putText(frame, 'Left Click', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    self.gesture_cooldown = 20

                elif fingers == [False, True, True, False, False] and self.gesture_cooldown == 0:
                    pyautogui.click(button='right')
                    cv2.putText(frame, 'Right Click', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    self.gesture_cooldown = 20

                elif fingers == [False, True, True, True, False] and self.scroll_cooldown == 0:
                    pyautogui.scroll(40)
                    cv2.putText(frame, 'Scroll Up', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    self.scroll_cooldown = 15

                elif fingers == [False, True, True, True, True] and self.scroll_cooldown == 0:
                    pyautogui.scroll(-40)
                    cv2.putText(frame, 'Scroll Down', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    self.scroll_cooldown = 15

                elif fingers == [True, False, False, False, False] and self.tab_cooldown == 0:
                    pyautogui.hotkey('alt', 'shift', 'tab')
                    cv2.putText(frame, '← Previous App', (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255), 2)
                    self.tab_cooldown = 30

                elif fingers == [False, False, False, False, True] and self.tab_cooldown == 0:
                    pyautogui.hotkey('alt', 'tab')
                    cv2.putText(frame, 'Next App →', (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255), 2)
                    self.tab_cooldown = 30

        return frame
