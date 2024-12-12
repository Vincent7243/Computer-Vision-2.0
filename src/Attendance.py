import cv2
import numpy as np
import os
import pickle

from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

BASE_PATH = os.path.abspath(os.path.dirname(__file__))
HAAR_CASCADE_PATH = os.path.join(BASE_PATH, "haarcascade_frontalface_default.xml")
FACES_PKL_PATH = os.path.join(BASE_PATH, "../data/faces_data.pkl")
NAMES_PKL_PATH = os.path.join(BASE_PATH, "../data/names.pkl")

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

try:
    with open(FACES_PKL_PATH, 'rb') as f:
        FACES = pickle.load(f)
    with open(NAMES_PKL_PATH, 'rb') as f:
        LABELS = pickle.load(f)
except FileNotFoundError:
    print("Không tìm thấy dữ liệu huấn luyện.")
    exit()

if len(FACES) == 0 or len(LABELS) == 0 or len(FACES) != len(LABELS):
    print("Dữ liệu huấn luyện không hợp lệ.")
    exit()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]  # Không chuyển sang grayscale
        resized_img = cv2.resize(crop_img, dsize=(50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, str(output[0]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
