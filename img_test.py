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
import sys

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y, w, h = point
    cv2.rectangle(image, (x, y - size[1]-10), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, (x,y-10), font, font_scale, (255, 255, 255), thickness)

img_size = 128
gray = False
img_path = "../test2.jpg"
image = cv2.imread(img_path)

#image = cv2.resize(image, (img_size,img_size))



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
