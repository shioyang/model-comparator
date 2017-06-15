from scipy.io import loadmat
import numpy as np
import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

debug = True
debug_roop = 100

num_classes = 10

# input image dimensions
img_rows, img_cols = 128, 128
input_shape = (img_rows, img_cols, 1)
# input_shape = (1, img_rows, img_cols)

data_dir = 'data'
 

def create_model():

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])
    
    # model.compile(loss='categorical_crossentropy',
    #           optimizer='adam',
    #           metrics=['accuracy'])
    
    return model


def load_data(mat_file_dir):
    x_img = []
    y_gender = []

    meta = loadmat(mat_file_dir)
    img_paths = meta["full_path"]
    genders   = meta["gender"]
 
    for i, path in enumerate(img_paths):
        if debug:
            print('i:', i)

        absPath = data_dir + '/' + path.strip()
        if debug:
            print('loading:', absPath)
            print('gender:', genders[0][i])

        img = load_img(absPath, target_size=(img_rows, img_cols))
        x_img.append( img_to_array(img) )

        y_gender.append( genders[0][i] )

        if debug and (i > debug_roop):
            break

    return np.array(x_img), np.array(y_gender)


# model = create_model()
# print(model)

x, y = load_data('modified_wiki.mat')
print('x:', x)
print('y:', y)
