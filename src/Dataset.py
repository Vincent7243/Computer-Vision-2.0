import cv2
import numpy as np
import os 
import pickle

video = cv2.VideoCapture(0)  # 0 = webcam camera
facedetect = cv2.CascadeClassifier("src/haarcascade_frontalface_default.xml")

face_data = []
i = 0 

name = input("Nhập tên của bạn:") 

# Tạo và cấu hình cửa sổ
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("frame", cv2.WND_PROP_TOPMOST, 1)

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w, :]
        resized_img = cv2.resize(crop_img, dsize=(50, 50))
        if len(face_data) <= 100 and i % 10 == 0:
            face_data.append(resized_img.flatten())  # Thêm dữ liệu vào danh sách
            cv2.putText(frame, str(len(face_data)), org=(50, 50), 
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, 
                        color=(50, 50, 255), thickness=1)  
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    cv2.imshow("frame", frame)
    i += 1
    
    if len(face_data) == 50:
        print("Đã thu thập đủ 50 dữ liệu khuôn mặt. Tắt camera.")
        break

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):  # Nhấn 'q' để thoát sớm
        print("Camera tắt bởi người dùng.")
        break

video.release()
cv2.destroyAllWindows()

#lưu khuôn mặt trong file pickle
face_data = np.array(face_data)
face_data = face_data.reshape(100, -1)

if 'names.pkl' not in os.listdir('data/'):
    names = [name]*100
    with open('data/names.pkl','wb') as f:
        pickle.dump(names,f)
else:
    with open('data/names.pkl','rb') as f:
        names = pickle.load(f)
    names = names + [name]*100

    with open('data/names.pkl','wb') as f:
        pickle.dump(names.f)

if 'face_data,pkl' not in os.listdir('data/'):
    with open('data/face_data,pkl','wb') as f:
        pickle.dump(face_data,f)
else:
    with open('data/face_data.pkl','rb') as f:
        faces = pickle.load(f)
    faces= np.append(faces, face_data,axis=0)
    with open('data/face_data,pkl','wb') as f:
        pickle.dump(faces,f)