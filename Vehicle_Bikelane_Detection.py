import cv2
import numpy as np
import pandas as np
import matplotlib.pyplot as plt
import keras
from keras import models
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import os
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from scipy.io import loadmat
import splitfolders
import glob
import shutil
from ultralytics import YOLO
import torch

#builds a new model using yolov8n.yaml
model = YOLO("yolov8n.yaml")

#used to train a new model
results = model.train(data="C:/Users/wwwsa/OneDrive/Desktop/ECE 396/Project_Code/Cars in Bike Lane.v1i.yolov8/data.yaml", epochs=4, batch=32 ,optimizer='Adam', lr0=.0001, workers=2)

#used to load custom trained model
#model = YOLO("C:/Users/wwwsa/OneDrive/Desktop/ECE 396/Project_Code/runs/detect/train/weights/last.pt")

#pretrained model
#model = YOLO('yolov8n.pt')


#Pic gets annotated with the pretrained model
'''pic = "C:/Users/wwwsa/OneDrive/Desktop/ECE 396/Project_Code/guy riding a bike pic.jpg"

cv2.imread(pic)

#cv2.resize(pic, (640,640))

results = model(pic)

print(results)

annotated_img = results[0].plot()

cv2.imshow("frame", annotated_img)
cv2.waitKey(0)

cv2.destroyAllWindows()'''

#names_dict = results[0].names

#probs = results[0].probs.tolist()

#print(results[0].boxes)


#Pretrained model (Working)
#Custom trained model is working, needs to be trained on more epochs and CHeck if boxes overlapping is an issue.
'''video_path = "C:/Users/wwwsa/OneDrive/Desktop/ECE 396/Project_Code/bicycle rider on road.mp4"

cap = cv2.VideoCapture(video_path)   #video_path

ret = True

while ret:
    ret, frame = cap.read()

    #cv2.resize(frame, (640,480))

    results = model.track(frame, persist=True)

    frame_ = results[0].plot()

    resized_frame_ = cv2.resize(frame_, (640,640))

    cv2.imshow('frame', resized_frame_)

    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
'''

#usign video to see if result is an annotated video... NOT WORKING, FIX
'''
video_path = "C:/Users/wwwsa/OneDrive/Desktop/ECE 396/Project_Code/bicycle rider on road.mp4"

Video_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(Video_out, fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (640, 640))

threshold = 0.1

class_name_dict = {0: 'vehicle', 1: 'bike-lane'}

#results = model.track(source='Bicycle rider on road.mp4', show=True, tracker = "bytetrack.yaml")

frame = cv2.resize(frame,(640,640))


while ret:
    cv2.imshow('before frame', frame)
    cv2.waitKey(10)

    results = model(frame)[0]
    print(results.tolist())

    for result in results:
        print("HERERER")
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)
    cv2.imshow('frame', frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()'''



#Test model 2
#This model cannot be used with our data because our data is set up differently
#The classes are not specified with the test, train images
#DATASET NEEDED FOR THIS
'''image_shape = (224,224)
training_data = "C:/Users/wwwsa/OneDrive/Desktop/ECE 396/Project_Code/Vehicle_detection_images_model2/train"
valid_data = "C:/Users/wwwsa/OneDrive/Desktop/ECE 396/Project_Code/Vehicle_detection_images_model2/test"

datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255) 

train_generator = datagen.flow_from_directory(
    training_data,
    shuffle=True,
    target_size=image_shape
)

valid_generator = datagen.flow_from_directory(
    valid_data,
    shuffle=True,
    target_size=image_shape
)

#print(train_generator.directory)
print(train_generator.class_indices)
print(valid_generator.class_indices)

model = keras.Sequential([
    keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation = 'relu', input_shape=(224,224,3)),
    keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation = 'relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation = 'relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation = 'softmax')
])

print(model.summary())

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=.0001),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

print(train_generator.samples)

Epochs = 10
Batch_size = 32
history = model.fit(train_generator,
                    steps_per_epoch= train_generator.samples // Batch_size,
                    epochs = Epochs,
                    validation_data= valid_generator,
                    validation_steps= valid_generator.samples // Batch_size,
                    verbose=1
)

model.save('car_bikelane_detection.model')'''

