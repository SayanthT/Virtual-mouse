import cv2
import mediapipe as mp
import numpy as np
import pyautogui

mp_drawing=mp.solutions.drawing_utils
mp_hands=mp.solutions.hands

SCREEN_WIDTH,SCREEN_HEIGHT=pyautogui.size()
pyautogui.PAUSE=0
pyautogui.FAILSAFE=False

hands=mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap=cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame=cv2.flip(frame, 1)

    image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results=hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x=int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * SCREEN_WIDTH)
            y=int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * SCREEN_HEIGHT)

            pyautogui.moveTo(x,y)

            thumb_tip=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2 + (thumb_tip.z - index_tip.z)**2)
            THRESHOLD=0.09

            if distance<THRESHOLD:
                pyautogui.click()

    cv2.imshow('Virtual Mouse', frame)

    if cv2.waitKey(10) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
