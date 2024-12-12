import cv2
import numpy as np
import os
import csv
import time
import pickle

from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


with open('data/names.pkl','rb') as w:
    LABELS = pickle.load(w)

with open('data/faces_data.pkl','rb') as f:
    FACES = pickle.load(f)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

imgbackground = cv2.imread("Background.png")

COL_NAMES=['NAME','TIME']


