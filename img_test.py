import keras
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.callbacks import LearningRateScheduler
import pickle
import os
import cv2
from keras.applications.mobilenetv2 import MobileNetV2
from keras.models import load_model
import numpy as np
#import sys

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y, w, h = point
    cv2.rectangle(image, (x, y - size[1]-10), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), thickness)
    #cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)
    cv2.putText(image, label, (x,y-10), font, font_scale, (255, 255, 255), thickness)

img_size = 128
gray = False
img_path = "../test2.jpg"
image = cv2.imread(img_path)

#image = cv2.resize(image, (img_size,img_size))

if gray:
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face = face_cascade.detectMultiScale(img,1.1)
    for x,y,w,h in  face:
        img = image[y:y+h,x:x+w]
    img = cv2.resize(img, (img_size,img_size))
    img = np.reshape(img, (1, img_size, img_size, 1))
else:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face = face_cascade.detectMultiScale(image)
    for x,y,w,h in  face:
        img = image[y:y+h,x:x+w]
    img = cv2.resize(img, (img_size,img_size))
    img = np.reshape(img, (1, img_size, img_size, 3))

weight_file = "./results_noaug_7465/weights.72-0.75-0.62.h5"
model = load_model(weight_file)
result = model.predict(img)

print(result)

race = np.argmax(result)

if race == 0:
    race_str = "Chinese"
elif race ==  1:
    race_str = "Japanese"
elif race == 2:
    race_str = "Korean"

print(race_str)

label = "{}".format(race_str)

for x,y,w,h in face:
    #cv2.rectangle(img, (x, y), (x+w,y+h), (255, 0, 0),3)
    draw_label(image, (x, y, w, h), label)
cv2.imwrite("im.png",image)
