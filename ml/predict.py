import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import argparse

img_rows, img_cols = 128, 128


def get_args():
    parser = argparse.ArgumentParser(description='Predict gender which the person\'s in the image is.')
    parser.add_argument('--input', '-i', type=str, required=True, help='path to input an image')
    return parser.parse_args()


args = get_args()
img_path = args.input

img = load_img(img_path, target_size=(img_rows, img_cols))
img_arr = img_to_array(img)
img_array = np.array([img_arr])


model = load_model('trained_model.h5')
model.summary()

print('********************')
print('*   0 for female   *')
print('*   1 for male     *')
print('********************')
results = model.predict(img_array, batch_size=img_array.shape[0])
for res in results:
    print('Result:')
    print('   Female:', res[0] * 100)
    print('   Male  :', res[1] * 100)

