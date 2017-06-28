import numpy as np
import argparse
import os
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

### Ignore tf warning: 'The TensorFlow library wasn't compiled to use...'
### Ref: https://github.com/tensorflow/tensorflow/issues/7778
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # Ignore tf warning: 'The TensorFlow library wasn't compiled to use...'

### Configuration
img_rows, img_cols = 128, 128
model_path = 'trained_model.h5'


def get_args():
    parser = argparse.ArgumentParser(description='Predict gender which the person\'s in the image is.')
    parser.add_argument('--input', '-i', type=str, required=True, help='path to input an image from the script path')
    return parser.parse_args()


args = get_args()
img_path = args.input

script_abs_path = os.path.abspath(os.path.dirname(__file__))


### Load Image for Prediction
print('')
print('### Loading img:', img_path)
img_full_path = script_abs_path + '/' + img_path
img = load_img(img_full_path, target_size=(img_rows, img_cols))
img_arr = img_to_array(img)
img_array = np.array([img_arr])
img_array = img_array.astype('float32')
img_array /= 255

### Load Trained Model
print('')
print('### Loading model:', model_path)
model_full_path = script_abs_path + '/' + model_path
model = load_model(model_full_path)
model.summary()


### Prediction
print('')
print('### Prediction')
print('********************')
print('*   0 for female   *')
print('*   1 for male     *')
print('********************')
results = model.predict(img_array, batch_size=img_array.shape[0])
for res in results:
    print('Result:')
    print('   Female:', res[0] * 100)
    print('   Male  :', res[1] * 100)
