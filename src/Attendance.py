import cv2
import numpy as np
import os
import pickle

from sklearn.neighbors import KNeighborsClassifier

# Đường dẫn đến các file cần thiết
BASE_PATH = os.path.abspath(os.path.dirname(__file__))
HAAR_CASCADE_PATH = os.path.join(BASE_PATH, "haarcascade_frontalface_default.xml")
FACES_PKL_PATH = os.path.join(BASE_PATH, "../data/faces_data.pkl")
NAMES_PKL_PATH = os.path.join(BASE_PATH, "../data/names.pkl")

# Khởi tạo video và nhận diện khuôn mặt
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

# Tải dữ liệu khuôn mặt và nhãn
try:
    with open(FACES_PKL_PATH, 'rb') as f:
        FACES = pickle.load(f)
    with open(NAMES_PKL_PATH, 'rb') as f:
        LABELS = pickle.load(f)
except FileNotFoundError:
    print("Không tìm thấy dữ liệu huấn luyện. Hãy chạy dataset.py trước.")
    exit()

# Kiểm tra dữ liệu huấn luyện hợp lệ
if len(FACES) == 0 or len(LABELS) == 0 or len(FACES) != len(LABELS):
    print("Dữ liệu huấn luyện không hợp lệ.")
    exit()

# Huấn luyện mô hình KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Bắt đầu nhận diện
while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:  # Nếu phát hiện khuôn mặt
        for (x, y, w, h) in faces:
            crop_img = frame[y:y + h, x:x + w]
            resized_img = cv2.resize(crop_img, dsize=(50, 50)).flatten().reshape(1, -1)
            
            try:
                output = knn.predict(resized_img)

                # Tách ID và tên từ chuỗi lưu trong names.pkl
                id_name = output[0]
                user_id, user_name = id_name.split(":")

                # Hiển thị ID và tên trên khung hình
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {user_id} - {user_name}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            except Exception as e:
                print(f"Lỗi nhận diện: {e}")
    else:
        # Nếu không phát hiện khuôn mặt, có thể hiển thị thông báo (tuỳ chọn)
        cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Attendance System", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
video.release()
cv2.destroyAllWindows()
