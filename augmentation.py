import cv2
import numpy as np
from PIL import Image
import os
import glob
import tensorflow as tf
import functions
import matplotlib.pyplot as plt

img_tr, img_tr_fn = functions.load_images("Originals")

path = os.path.abspath(os.getcwd())
folder = path.rpartition("/")
print(folder)

# print(len(img_tr))
# print(len(img_tr_fn))

def augment(img_tr, img_tr_fn):
    k=0
    img_aug = []
    img_aug_fn = []
    while k < 2: 
        j=0; i=0
        for image in img_tr:
            image_orig = image.copy()

            # brightness
            if tf.random.uniform(shape=(), minval=0, maxval=1, dtype=tf.dtypes.float32).numpy() < 0.5:
                bright = np.ones(image.shape , dtype="uint8") * tf.random.uniform(shape=(), minval=0, maxval=100, dtype=tf.dtypes.int32).numpy()
                image = cv2.add(image,bright)
            elif tf.random.uniform(shape=(), minval=0, maxval=1, dtype=tf.dtypes.float32).numpy() < 0.5:
                bright = np.ones(image.shape , dtype="uint8") * tf.random.uniform(shape=(), minval=0, maxval=100, dtype=tf.dtypes.int32).numpy()
                image = cv2.subtract(image,bright)
            
            # zoom
            if tf.random.uniform(shape=(), minval=0, maxval=1, dtype=tf.dtypes.float32).numpy() < 0.7:
                image = zoom_center(image, tf.random.uniform(shape=(), minval=1, maxval=1.5, dtype=tf.dtypes.float32).numpy())
            
            # # gaussian filter
            # if tf.random.uniform(shape=(), minval=0, maxval=1, dtype=tf.dtypes.float32).numpy() < 0.5:
            #     mean=0
            #     st=0.7
            #     gauss = np.random.normal(mean,st,image.shape)
            #     gauss = gauss.astype('uint8')
            #     image = cv2.add(image,gauss)

            # blur filter
            if tf.random.uniform(shape=(), minval=0, maxval=1, dtype=tf.dtypes.float32).numpy() < 0.5:
                fsize = tf.random.uniform(shape=(), minval=2, maxval=6, dtype=tf.dtypes.int32).numpy()
                image = cv2.blur(image,(fsize,fsize))
                        
            # sharpening filter
            if tf.random.uniform(shape=(), minval=0, maxval=1, dtype=tf.dtypes.float32).numpy() < 0.5:
                sharpening = np.array([ [-1,-1,-1],
                                        [-1,10,-1],
                                        [-1,-1,-1] ])
                image = cv2.filter2D(image,-1,sharpening)

            if not np.array_equal(image, image_orig):
                img_aug.append(image)
                img_aug_fn.append(img_tr_fn[i])
                j+=1
            i+=1
        k+=1
    print(len(img_aug))
    print(len(img_aug_fn))
    print(img_aug_fn)

    return img_aug, img_aug_fn


def zoom_center(img, zoom_factor):

    y_size = img.shape[0]
    x_size = img.shape[1]
    
    # define new boundaries
    x1 = int(0.5*x_size*(1-1/zoom_factor))
    x2 = int(x_size-0.5*x_size*(1-1/zoom_factor))
    y1 = int(0.5*y_size*(1-1/zoom_factor))
    y2 = int(y_size-0.5*y_size*(1-1/zoom_factor))

    # first crop image then scale
    img_cropped = img[y1:y2,x1:x2]
    
    return cv2.resize(img_cropped, None, fx=zoom_factor, fy=zoom_factor)


def save_images(folder, img_array, img_array_fn):
    script_dir = os.path.abspath(os.path.dirname(__file__))

    if not os.path.isdir(folder):
        os.makedirs(folder)

    results_dir = os.path.join(script_dir, folder) + '/'

    for i in range(len(img_array)):
        plt.rcParams["savefig.directory"] = os.chdir(os.path.dirname(results_dir))
        im_rgb = cv2.cvtColor(img_array[i], cv2.COLOR_BGR2RGB)
        np.array(im_rgb,np.int32)
        plt.imshow(im_rgb)
        plt.axis('off')
        plt.savefig(str(img_aug_fn[i]) + "_aug" + str(i), bbox_inches='tight', pad_inches=-0.1)     
          


img_aug, img_aug_fn = augment(img_tr, img_tr_fn)

save_images("Augmented v2", img_aug, img_aug_fn)

print("FINAL: " + str(img_aug_fn))
print("FINAL: " + str(len(img_aug)))
print("FINAL: " + str(len(img_aug_fn)))