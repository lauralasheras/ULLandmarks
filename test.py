import tensorflow as tf
import time
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from torch import rand
import functions

# plt.imshow(images[1].astype(np.uint8))
# plt.show()

# n = 2

# f, axarr = plt.subplots(1,2, dpi=100)
# axarr[0].imshow(orig[n])
# axarr[0].set_title('original')
# axarr[1].imshow(images[n])
# axarr[1].set_title('after reshape')

# plt.show()

def label_images(images_fn):
    labels = []
    for i in images_fn:
        name = i.rpartition("_")
        index = images_fn.tolist().index(i)
        if name[2].isnumeric():
            l = name[0]
        else:
            l = name[0].rpartition("_")[0]
        # print("NAME:" + str(name))
        # print("NAME2:" + str(name))
        print("LABEL: " + str(l))
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

train, train_fn = functions.load_images("Augmented")

train_fn_l = label_images(train_fn)

# print("Train_fn: " + str(train_fn))
# print("Train_fn_l: " + str(train_fn_l))
# print("Len train_fn: " + str(len(train_fn)))
# print("Len train_fn_l: " + str(len(train_fn_l)))

plt.imshow(train[433])

print(train[433].shape)


plt.show()