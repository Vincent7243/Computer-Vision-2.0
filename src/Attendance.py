import cv2
import numpy as np
import os
import csv
import time
import pickle

from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("src/haarcascade_frontalface_default.xml")

# Kiểm tra và đọc dữ liệu
try:
    with open('data/names.pkl', 'rb') as w:
        LABELS = pickle.load(w)
except EOFError:
    print("Lỗi: Tệp 'names.pkl' trống hoặc hỏng.")
    LABELS = []

try:
    with open('data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)
except EOFError:
    print("Lỗi: Tệp 'faces_data.pkl' trống hoặc hỏng.")
    FACES = []

# Kiểm tra nếu dữ liệu trống
if len(LABELS) == 0 or len(FACES) == 0:
    print("Lỗi: Dữ liệu huấn luyện không hợp lệ.")
    exit()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

imgbackground = cv2.imread("Background.png")

COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Đã sửa lại từ cv2.COLOR_BG2GRAY
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, dsize=(50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, text=str(output[0]), org=(x + w, y + h), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(50, 50, 255), thickness=1)
        attendance = [str(output[0]), str(timestamp)]
    
    imgbackground[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("frame", imgbackground)

    k = cv2.waitKey(1)
    if k == ord('o'):
        time.sleep(5)
        if exist:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
        else:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)

    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
