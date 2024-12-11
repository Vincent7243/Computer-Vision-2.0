import cv2
import numpy as np
import os 
import pickle

video = cv2.VideoCapture(0) # 0 = webcam camera
facedetect = cv2.CascadeClassifier("src/haarcascade_frontalface_default.xml")

face_data = []

i =0 

name = input("Nhập tên của bạn:") 