import tensorflow as tf
import time
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from torch import rand

img_dim = 1000
directory = "Augmented v2"
no_of_epochs = 100

def load_images(folder):
    images = []
    images_fn = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            resized_img = cv2.resize(img, (img_dim, img_dim))
            gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)            
            images.append(gray)
            images_fn.append(os.path.splitext(filename)[0])
            
    images_np = np.array(images)
    images_fn_np = np.array(images_fn)
    
    return images_np, images_fn_np


def label_images_v(images_fn):
    labels = []
    for i in images_fn:
        name = i.split("_")
        index = images_fn.tolist().index(i)
        if name[0] == 'a':
            labels.insert(index, 0)
        elif name[0] == 'i':
            labels.insert(index, 1)
        elif name[0] == 'u':
            labels.insert(index, 2)
        elif name[0] == 'e':
            labels.insert(index, 3)
        elif name[0] == 'o':
            labels.insert(index, 4)
    labels_np = np.array(labels)
    return labels_np

def label_images(images_fn):
    labels = []
    for i in images_fn:
        name = i.rpartition("_")
        index = images_fn.tolist().index(i)
        if name[2].isnumeric():
            l = name[0]
        else:
            l = name[0].rpartition("_")[0]
        if l == 'Main_Building':
            labels.insert(index, 0)
        elif l == 'Statue':
            labels.insert(index, 1)
        elif l == 'Statue_Main_Building':
            labels.insert(index, 2)
        elif l == 'Statue_Foundation_Building':
            labels.insert(index, 3)
        elif l == 'Foundation_Building':
            labels.insert(index, 4)
        elif l == 'Sports_Arena':
            labels.insert(index, 5)
        elif l == 'Business_school':
            labels.insert(index, 6)
        elif l == 'Flag_Poles':
            labels.insert(index, 7)
        elif l == 'Plassey_House':
            labels.insert(index, 8)
        elif l == 'Bridge':
            labels.insert(index, 9)
        elif l == 'Bridge_Plassey_House':
            labels.insert(index, 10)
        elif l == 'Bridge_Irish_Academy':
            labels.insert(index, 11)
        elif l == 'Bridge_Health_Sciences':
            labels.insert(index, 12)
        elif l == 'Bridge_Health_Sciences_Irish_Academy':
            labels.insert(index, 13)
        elif l == 'Health_Sciences':
            labels.insert(index, 14)
        elif l == 'Irish_Academy':
            labels.insert(index, 15)
        elif l == 'Health_Sciences_Irish_Academy':
            labels.insert(index, 16)
    labels_np = np.array(labels)
    return labels_np

def test(train, train_fn):
    t = round(len(train)*0.8)
    print("T: " + str(t))

    test = np.zeros([len(train)-t, img_dim, img_dim], dtype=np.int32)
    test_fn = np.zeros([len(train)-t], dtype=np.int32)
    i=0
    while len(train) > t:
        rand_n = tf.random.uniform(shape=(), minval=0, maxval=len(train), dtype=tf.dtypes.int32).numpy()
        test[i] = train[rand_n].copy()
        test_fn[i] = train_fn[rand_n].copy()
        train = np.delete(train, rand_n, axis=0)
        train_fn = np.delete(train_fn, rand_n, axis=0)
        i+=1
    return test, test_fn