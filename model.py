# ------------------------------------------------------------------
# Filename:    TensorFlow_mnist.py
# ------------------------------------------------------------------
# File description:
# Python and TensorFlow image classification using the MNIST dataset.
# ------------------------------------------------------------------

# ------------------------------------------------------
# Modules to import
# ------------------------------------------------------

from cv2 import imshow
import tensorflow as tf
import time
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from torch import rand
import os
import functions
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ------------------------------------------------------
# def load_dataset(dataset)
# ------------------------------------------------------

def load_dataset():

    train, train_fn = functions.load_images(functions.directory)
    print("Dataset size: " + str(len(train)))
    print("Labels size antes: " + str(len(train_fn))) 
    train_fn = functions.label_images(train_fn)
    print("Labels size: " + str(len(train_fn))) 
    test, test_fn = functions.test(train, train_fn)

    train[:], test[:] = [x / 255.0 for x in train], [y / 255.0 for y in test]

    return train, train_fn, test, test_fn

# ------------------------------------------------------
# def create_model()
# ------------------------------------------------------

def create_model():

    model = tf.keras.models.Sequential(
        [tf.keras.layers.Flatten(input_shape=(functions.img_dim, functions.img_dim)),
         # tf.keras.layers.Dense(512, activation='relu'),
         tf.keras.layers.Dense(units=2048, activation='relu'),
         tf.keras.layers.Dropout(0.2),
         tf.keras.layers.Dense(17, activation='softmax')]
        )

    return model  


# ------------------------------------------------------
# def compile_model(model)
# ------------------------------------------------------

def compile_model(model):
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
        )


# ------------------------------------------------------
# def fit_model(model, x_train, x_train,x_test, y_test, no_of_epochs)
# ------------------------------------------------------

def fit_model(model, x_train, y_train, x_test, y_test, no_of_epochs):

    history_callback = model.fit(
        x=x_train,
        y=y_train,
        epochs=no_of_epochs,
        validation_data=(x_test, y_test),
        )

    return history_callback


# ------------------------------------------------------
# def image_prediction(model, image)
# ------------------------------------------------------

def image_prediction(model, image):

    print('-- Image to predict')
    print('Shape of image ', image.shape)

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(image.reshape(1, functions.img_dim, functions.img_dim))

    print(predictions)

    print('-- Predicted image')
    print('Most probable image ', tf.argmax(predictions[0]))


# ------------------------------------------------------
# def model_save(model, model_dir)
# ------------------------------------------------------

def model_save(model, model_dir):

    tf.keras.models.save_model(
        model,
        model_dir,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        )


# ------------------------------------------------------
# def model_load(model_dir)
# ------------------------------------------------------

def model_load(model_dir):

    reconstructed_model = tf.keras.models.load_model(
        model_dir,
        custom_objects=None,
        compile=True,
        )

    return reconstructed_model


# ------------------------------------------------------
# def plot_image(image_to_predict)
# ------------------------------------------------------

def plot_image(image_to_predict):

    fig, axs = plt.subplots(1, 3)
    plt.suptitle('Image to predict')

    axs[0].imshow(image_to_predict)
    axs[0].set_title('Original image', fontsize=10)
    axs[0].set_xlabel('x pixel', fontsize=10)
    axs[0].set_ylabel('y pixel', fontsize=10)

    axs[1].imshow(image_to_predict, cmap=plt.get_cmap('gray'))
    axs[1].set_title('CMAP grayscale image', fontsize=10)
    axs[1].set_xlabel('x pixel', fontsize=10)
    axs[1].set_ylabel('y pixel', fontsize=10)
    
    axs[2].imshow(image_to_predict, cmap=plt.get_cmap('binary'))
    axs[2].set_title('CMAP binary image', fontsize=10)
    axs[2].set_xlabel('x pixel', fontsize=10)
    axs[2].set_ylabel('y pixel', fontsize=10)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------
# def plot_statistics(history_callback)
# ------------------------------------------------------

def plot_statistics(history_callback, no_of_epochs):

    loss_history = history_callback.history['loss']
    accuracy_history = history_callback.history['accuracy']
    val_loss_history = history_callback.history['val_loss']
    val_accuracy_history = history_callback.history['val_accuracy']

    epochs = tf.linspace(1, no_of_epochs, no_of_epochs)

    fig, axs = plt.subplots(1, 4, figsize=(10, 5))

    axs[0].plot(epochs, loss_history)
    axs[0].set_title('Training loss', fontsize=10)
    axs[0].set_xlabel('Epoch', fontsize=10)
    axs[0].set_ylabel('Loss', fontsize=10)
    axs[0].grid(True)
    axs[0].set_facecolor('palegreen')

    axs[1].plot(epochs, accuracy_history)
    axs[1].set_title('Training accuracy', fontsize=10)
    axs[1].set_xlabel('Epoch', fontsize=10)
    axs[1].set_ylabel('Accuracy', fontsize=10)
    axs[1].grid(True)
    axs[1].set_facecolor('palegreen')

    axs[2].plot(epochs, val_loss_history)
    axs[2].set_title('Validation loss', fontsize=10)
    axs[2].set_xlabel('Epoch', fontsize=10)
    axs[2].set_ylabel('Loss', fontsize=10)
    axs[2].grid(True)
    axs[2].set_facecolor('gainsboro')

    axs[3].plot(epochs, val_accuracy_history)
    axs[3].set_title('Validation accuracy', fontsize=10)
    axs[3].set_xlabel('Epoch', fontsize=10)
    axs[3].set_ylabel('Accuracy', fontsize=10)
    axs[3].grid(True)
    axs[3].set_facecolor('gainsboro')

    plt.suptitle('Model fitting statistics')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    plt.show()


# ------------------------------------------------------
# def main()
# ------------------------------------------------------

def main():

    # ------------------------------------------------------
    # -- Start of script run actions
    # ------------------------------------------------------

    print('----------------------------------------------------')
    print('-- Start script run ' + str(time.strftime('%c')))
    print('----------------------------------------------------\n')
    print('-- Python version     : ' + str(sys.version))
    print('-- TensorFlow version : ' + str(tf.__version__))
    print('-- Matplotlib version : ' + str(mpl.__version__))

    dataset = tf.keras.datasets.mnist


    print('>> Number of epochs = ', functions.no_of_epochs)

    # ------------------------------------------------------
    # -- Main script run actions
    # ------------------------------------------------------

    print('\n----------------------------------------------------------')
    print('-- 1. Load the dataset')
    print('----------------------------------------------------------\n')

    (x_train, y_train, x_test, y_test) = load_dataset()

    print('\n----------------------------------------------------------')
    print('-- 2. Create the model')
    print('----------------------------------------------------------\n')

    model = create_model()

    print('\n----------------------------------------------------------')
    print('-- 3. Compile the model')
    print('----------------------------------------------------------\n')

    compile_model(model)

    print('\n----------------------------------------------------------')
    print('-- 4. Train the model using the training data')
    print('----------------------------------------------------------\n')

    history_callback = fit_model(model, x_train, y_train, x_test, y_test, functions.no_of_epochs)

    loss_history = history_callback.history['loss']
    accuracy_history = history_callback.history['accuracy']
    val_loss_history = history_callback.history['val_loss']
    val_accuracy_history = history_callback.history['val_accuracy']

    weights = model.get_weights()

    print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(model.summary())
    print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(weights)
    print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('loss         = ', loss_history)
    print('accuracy     = ', accuracy_history)
    print('avg accuracy     = ', sum(accuracy_history)/len(accuracy_history)*100)
    print('val_loss     = ', val_loss_history)
    print('val_accuracy = ', val_accuracy_history)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    print('\n----------------------------------------------------------')
    print('-- 5. Evaluate the model using the test image set')
    print('----------------------------------------------------------\n')

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

    print('Test Loss:     ', test_loss)
    print('Test Accuracy: ', test_acc)

    print('\n----------------------------------------------------------')
    print('-- 6. Predict image from the test image set using the model')
    print('----------------------------------------------------------\n')

    test_image_id = tf.random.uniform((), minval=0, maxval=x_test.shape[0], dtype=tf.int32)
    image_to_predict = x_test[test_image_id]
    image_to_predict_label = y_test[test_image_id]
    # imshow(image_to_predict)

    print('** Image to predict1 is x_test[', test_image_id.numpy(), '] **')
    print('** This image has a label y_test[', image_to_predict_label, '] **')

    image_prediction(model, image_to_predict)
    print("--- %.2f seconds ---" % (time.time() - start_time))


    print('\n----------------------------------------------------------')
    print('-- 7. Plot the image to predict')
    print('----------------------------------------------------------\n')

    print('Image to plot is x_test[', test_image_id.numpy(), ']')

    plot_image(image_to_predict)

    print('\n----------------------------------------------------------')
    print('-- 7. Plot the loss and accuracy')
    print('----------------------------------------------------------\n')

    plot_statistics(history_callback, functions.no_of_epochs)

    # ------------------------------------------------------
    # -- End of script run actions
    # ------------------------------------------------------

    print('----------------------------------------------------')
    print('-- End script run ' + str(time.strftime('%c')))
    print('----------------------------------------------------\n')


# ------------------------------------------------------
# Run only if source file is run as the main script
# ------------------------------------------------------

if __name__ == '__main__':
    start_time = time.time()
    main()
# ------------------------------------------------------------------
# End of script
# ------------------------------------------------------------------
