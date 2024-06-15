import cv2
import numpy as np
import pygame
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from scipy.spatial import distance as dist
from threading import Thread

def start_alarm(sound):
    """Play the alarm sound"""
    pygame.mixer.init()
    pygame.mixer.music.load('alarmsound.mp3')
    pygame.mixer.music.play()

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_yawn(mouth):
    if len(mouth) < 6:
        return False
    A = dist.euclidean(mouth[0], mouth[3])
    B = dist.euclidean(mouth[4], mouth[1])
    mar = B / A
    return mar > 0.6

classes = ['Closed', 'Open']
import os

face_cascade_path = os.path.abspath('haarcascade_frontalface_default.xml')
left_eye_cascade_path = os.path.abspath('haarcascade_lefteye_2splits.xml')
right_eye_cascade_path = os.path.abspath('haarcascade_righteye_2splits.xml')
mouth_cascade_path = os.path.abspath('haarcascade_mouth.xml')

face_cascade = cv2.CascadeClassifier(face_cascade_path)
left_eye_cascade = cv2.CascadeClassifier(left_eye_cascade_path)
right_eye_cascade = cv2.CascadeClassifier(right_eye_cascade_path)
mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)

cap = cv2.VideoCapture(0)
model = load_model("driver-fatigue.h5")
count = 0
alarm_on = False
alarm_sound = "alarmsousnd.mp3"
status1 = ''
status2 = ''
status_mouth = ''

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame from the camera.")
        break

    height = frame.shape[0]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        left_eye = left_eye_cascade.detectMultiScale(roi_gray)
        right_eye = right_eye_cascade.detectMultiScale(roi_gray)
        mouth = mouth_cascade.detectMultiScale(roi_gray)
        
        for (x1, y1, w1, h1) in left_eye:
            cv2.rectangle(roi_color, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 1)
            eye1 = roi_color[y1:y1+h1, x1:x1+w1]
            eye1 = cv2.resize(eye1, (145, 145))
            eye1 = eye1.astype('float') / 255.0
            eye1 = img_to_array(eye1)
            eye1 = np.expand_dims(eye1, axis=0)
            pred1 = model.predict(eye1)
            status1=np.argmax(pred1)
            break

        for (x2, y2, w2, h2) in right_eye:
            cv2.rectangle(roi_color, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 1)
            eye2 = roi_color[y2:y2 + h2, x2:x2 + w2]
            eye2 = cv2.resize(eye2, (145, 145))
            eye2 = eye2.astype('float') / 255.0
            eye2 = img_to_array(eye2)
            
            eye2 = np.expand_dims(eye2, axis=0)
            pred2 = model.predict(eye2)
            status2=np.argmax(pred2)
            break

        for (mx, my, mw, mh) in mouth:
            cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 0, 255), 1)
            mouth_aspect_ratio = detect_yawn([(mx, my), (mx+mw//2, my+mh//2), (mx+mw, my)])
            if mouth_aspect_ratio:
                status_mouth = 'Yawning'
            else:
                status_mouth = 'Not Yawning'
            break

        if (status1 == 2 and status2 == 2) or status_mouth == 'Yawning':
            count += 1
            cv2.putText(frame, "Eyes Closed or Yawning, Frame count: " + str(count), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
            if count >= 2:
                cv2.putText(frame, "Drowsiness Alert!!!", (100, height-20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                if not alarm_on:
                    alarm_on = True
                    t = Thread(target=start_alarm, args=(alarm_sound,))
                    t.daemon = True
                    t.start()
        else:
            cv2.putText(frame,"Eyes Open Not Yawning", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            count = 0
            alarm_on = False

    cv2.imshow("Drowsiness Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
