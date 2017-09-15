from scipy.io import loadmat
import numpy as np
import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

### For debug
DEBUG = False
DEBUG_VERBOSE = False   # Ignore, if DEBUG is False

AUGMENTATION = True

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


def create_model01():

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


from keras.applications.vgg16 import VGG16
from keras.layers import Input
from keras.models import Model
from keras import optimizers

def create_model02(nb_label):
    # Make tensor
    input_tensor = Input(shape=input_shape)

    # Assign input_tensor, because output_shape becomes None without it.
    vgg16_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    top_model = Sequential()
    top_model.add( Flatten(input_shape=vgg16_model.output_shape[1:]) )
    top_model.add( Dense(256, activation='relu') )
    top_model.add( Dropout(0.5) )
    top_model.add( Dense(nb_label, activation='softmax') )

    # Connect vgg16 and top_model
    model = Model(input=vgg16_model.input, output=top_model(vgg16_model.output))

    # Freeze layers which are just before last convolution layer.
    for layer in model.layers[:15]:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])

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
# model = create_model01()
model = create_model02(num_classes)
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
#        Preparation         #
#============================#
if AUGMENTATION:
    data_gen = ImageDataGenerator(
        featurewise_center=False,             # Set input mean to 0 over the dataset, feature-wise.
        samplewise_center=False,              # Set each sample mean to 0.
        featurewise_std_normalization=False,  # Divide inputs by std of the dataset, feature-wise.
        samplewise_std_normalization=False,   # Divide each input by its std.
        zca_whitening=False,                  # Apply ZCA whitening.
        rotation_range=10,                    # Int. Degree range for random rotations.
        width_shift_range=0.3,                # Float (fraction of total width). Range for random horizontal shifts.
        height_shift_range=0.2,               # Float (fraction of total height). Range for random vertical shifts.
        horizontal_flip=True,                 # Randomly flip inputs horizontally.
        vertical_flip=False,                  # Randomly flip inputs vertically.
        # -----
        # zca_epsilon=1e-6,                     # epsilon for ZCA whitening. Default is 1e-6.
        shear_range=0.,                       # Float. Shear Intensity (Shear angle in counter-clockwise direction as radians)
        zoom_range=0.2,                       # Float or [lower, upper]. Range for random zoom.
                                              # If a float,  [lower, upper] = [1-zoom_range, 1+zoom_range].
        channel_shift_range=0.,               # Float. Range for random channel shifts.
        fill_mode='nearest',                  # One of {"constant", "nearest", "reflect" or "wrap"}.
                                              # Points outside the boundaries of the input are filled according to the given mode.
        cval=0.,                              # Float or Int. Value used for points outside the boundaries when fill_mode = "constant".
        rescale=None,                         # rescaling factor. Defaults to None. If None or 0, no rescaling is applied,
                                              # otherwise we multiply the data by the value provided (before applying any other transformation).
        preprocessing_function=None           # function that will be implied on each input. The function will run before any other modification on it.
                                              # The function should take one argument: one image (Numpy tensor with rank 3), and should output a Numpy tensor with the same shape.
    )

    # Split data to train and test
    #   train : test = 8 : 2
    num_split = int(round((len(x) / 10) * 8))

    x_train = x[ :num_split]
    y_train = y[ :num_split]
    x_test  = x[num_split: ]
    y_test  = y[num_split: ]

    # Need for featurewise_center, featurewise_std_normalization, and zca_whitening
    data_gen.fit(x_train)


#============================#
#          Training          #
#============================#
epochs = 50
batch_size = 32


### For TensorBoard
# tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir) # Histogram off, because not support when using generator.
callbacks = [tb_callback]
###


if AUGMENTATION:
    model.fit_generator(
        data_gen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=x_train.shape[0],
        # samples_per_epoch=x_train.shape[0],
        epochs=epochs,
        # nb_epoch=epochs,
        validation_data=data_gen.flow(x_test, y_test),
        validation_steps=x_test.shape[0],
        # nb_val_samples=x_test.shape[0],
        callbacks=callbacks
    )
else:
    model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=0.2,
                 callbacks=callbacks ) # for TensorBoard

model.save('trained_model.h5')


### For TensorBoard
tensorflow_backend.set_session(old_session)
###
