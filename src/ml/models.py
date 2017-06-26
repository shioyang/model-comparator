from scipy.io import loadmat
import numpy as np
import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical

### For debug
DEBUG = False
DEBUG_VERBOSE = False   # Ignore, if DEBUG is False

### input image dimensions
img_rows, img_cols = 128, 128
rgb = 3
input_shape = (img_rows, img_cols, rgb)   # data format: channels last
# input_shape = (rgb, img_rows, img_cols)   # data format: channels first

DATA_DIR = 'data'
DATA_COUNT = 2000
num_classes = 2


### For TensorBoard
from keras.backend import tensorflow_backend
import tensorflow as tf
log_dir = './tflog/'
old_session = tensorflow_backend.get_session()
session = tf.Session()
tensorflow_backend.set_session(session)
tensorflow_backend.set_learning_phase(1)
###


def create_model():

    model = Sequential()

    # input: (img_rows x img_cols) images with 3 channels -> (img_rows, img_cols, 3) tensors.
    # This applies 32 convolution filters of size 3x3 each.
    model.add( Conv2D(32, kernel_size=(3, 3),               # Output Shape           Param #
                  activation='relu',                        # -------------------------------
                  input_shape=input_shape) )                # (None, 126, 126, 32)       896
    model.add( Conv2D(64, (3, 3), activation='relu') )      # (None, 124, 124, 64)     18496
    model.add( MaxPooling2D(pool_size=(2, 2)) )             # (None,  62,  62, 64)         0
    model.add( Dropout(0.25) )                              # (None,  62,  62, 64)         0
    model.add( Flatten() )                                  # (None,       246016)         0
    model.add( Dense(128, activation='relu') )              # (None,          128)  31490176
    model.add( Dropout(0.5) )                               # (None,          128)         0
    model.add( Dense(num_classes, activation='softmax') )   # (None,            2)       258

    model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])

    # model.compile(loss='categorical_crossentropy',
    #           optimizer='adam',
    #           metrics=['accuracy'])

    return model


def appendImg(img_paths, genders, count, x_img, y_gender):
    for i, path in enumerate(img_paths):
        if DEBUG and DEBUG_VERBOSE:
            print('i:', i)

        absPath = DATA_DIR + '/' + path.strip()
        if DEBUG and DEBUG_VERBOSE:
            print('loading:', absPath)
            print('gender:', genders[0][i])

        img = load_img(absPath, target_size=(img_rows, img_cols))
        x_img.append( img_to_array(img) )

        y_gender.append( genders[0][i] )

        if i > count:
            break


def load_data(female_mat_file, male_mat_file):
    female_meta = loadmat(female_mat_file)
    male_meta   = loadmat(  male_mat_file)

    female_img_paths = female_meta["full_path"]
    female_genders   = female_meta["gender"]
    male_img_paths = male_meta["full_path"]
    male_genders   = male_meta["gender"]
 
    x_img = []
    y_gender = []
    appendImg(female_img_paths, female_genders, DATA_COUNT / 2, x_img, y_gender)
    appendImg(  male_img_paths,   male_genders, DATA_COUNT / 2, x_img, y_gender)

    return np.array(x_img), np.array(y_gender)


#============================#
#          Modeling          #
#============================#
model = create_model()
model.summary()

x, y = load_data('female_wiki.mat', 'male_wiki.mat')
x = x.astype('float32')
x /= 255   # Normalize to values between 0..1

#  ' while using as loss `categorical_crossentropy`. '
# ValueError: You are passing a target array of shape (102, 1) while using as loss `categorical_crossentropy`. `categorical_crossentropy` expects targets
#  to be binary matrices (1s and 0s) of shape (samples, classes). If your targets are integer classes, you can convert them to the expected format via:
# ```
# from keras.utils.np_utils import to_categorical
# y_binary = to_categorical(y_int)
# ```
# 
# Alternatively, you can use the loss function `sparse_categorical_crossentropy` instead, which does expect integer targets.
y = to_categorical(y, num_classes=num_classes)   # Converts a class vector (integers) to binary class matrix.

if DEBUG and DEBUG_VERBOSE:
    print('x:', x)
    print('y:', y)
print('y:', y)

if DEBUG:
    countF = 0
    countM = 0
    for y0 in y:
        if y0[0] == 1:
            countF += 1
        if y0[1] == 1:
            countM += 1
    print('data:', countF)
    print('   Female:', countF)
    print('   Male  :', countM)

    exit()

#============================#
#          Training          #
#============================#
epochs = 10


### For TensorBoard
tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
callbacks = [tb_callback]
###


model.fit(x, y, batch_size=32, epochs=epochs, validation_split=0.2,
             callbacks=callbacks ) # for TensorBoard

model.save('trained_model.h5')


### For TensorBoard
tensorflow_backend.set_session(old_session)
###
