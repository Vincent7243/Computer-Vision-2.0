import cv2
import numpy as np
import os
import pickle

BASE_PATH = os.path.abspath(os.path.dirname(__file__))
FACES_PKL_PATH = os.path.join(BASE_PATH, "../data/faces_data.pkl")
NAMES_PKL_PATH = os.path.join(BASE_PATH, "../data/names.pkl")
HAAR_CASCADE_PATH = os.path.join(BASE_PATH, "haarcascade_frontalface_default.xml")

os.makedirs(os.path.dirname(FACES_PKL_PATH), exist_ok=True)

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

face_data = []
i = 0

name = input("Nhập tên của bạn: ")

while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]
        crop_img = crop_img[:, :, :3]  # Chỉ giữ kênh RGB
        resized_img = cv2.resize(crop_img, dsize=(50, 50))
        if len(face_data) <= 100 and i % 10 == 0:
            face_data.append(resized_img.flatten())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)

    cv2.imshow("Frame", frame)
    i += 1
    if len(face_data) == 100:
        print("Đã thu thập đủ dữ liệu.")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

face_data = np.array(face_data)

if not os.path.exists(FACES_PKL_PATH):
    with open(FACES_PKL_PATH, 'wb') as f:
        pickle.dump(face_data, f)
else:
    with open(FACES_PKL_PATH, 'rb') as f:
        old_faces = pickle.load(f)
    if old_faces.shape[1] != face_data.shape[1]:
        print("Lỗi: Định dạng dữ liệu cũ không khớp. Hãy xóa tệp faces_data.pkl và thử lại.")
        exit()
    face_data = np.append(old_faces, face_data, axis=0)
    with open(FACES_PKL_PATH, 'wb') as f:
        pickle.dump(face_data, f)

if not os.path.exists(NAMES_PKL_PATH):
    names = [name] * 100
    with open(NAMES_PKL_PATH, 'wb') as f:
        pickle.dump(names, f)
else:
    with open(NAMES_PKL_PATH, 'rb') as f:
        old_names = pickle.load(f)
    names = old_names + [name] * 100
    with open(NAMES_PKL_PATH, 'wb') as f:
        pickle.dump(names, f)
