import numpy as np
from keras.preprocessing.image import load_img, list_pictures, img_to_array

def load_img_asarray(data_dir, ext):
    '''
    Import images from the directory as array, and append it to container array, X:
      2 pictures as array
         [
             array([...], dtype=float32),
             array([...], dtype=float32)
         ]
    -----------------------------
    [array([[[ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            ..., 
            [ 0.,  1.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  1.,  0.]],
    
           [[ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            ..., 
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.]]], dtype=float32), array([[[   0.,    0.,    0.],
            [   0.,    0.,    0.],
            [   0.,    0.,    0.],
            ..., 
            [   0.,    0.,    0.],
            [   0.,    0.,    0.],
            [   0.,    0.,    0.]],
    
           [[   1.,    0.,    0.],
            [   1.,    0.,    0.],
            [   6.,    2.,    1.],
            ..., 
            [  65.,   94.,   74.],
            [  66.,   95.,   73.],
            [  65.,   94.,   72.]]], dtype=float32)]
    '''
    X = []
    for pic in list_pictures(data_dir, ext):
        img = img_to_array( load_img(pic) )
        X.append(img)
    
    '''
    Convert array() to []:
    -----------------------------
    [[[[   0.    0.    0.]
       [   0.    0.    0.]
       [   0.    0.    0.]
       ..., 
       [   0.    1.    0.]
       [   0.    1.    0.]
       [   0.    1.    0.]]
    
      [[   0.    0.    0.]
       [   0.    0.    0.]
       [   0.    0.    0.]
       ..., 
       [   0.    0.    0.]
       [   0.    0.    0.]
       [   0.    0.    0.]]]
    
     [[[   0.    0.    0.]
       [   0.    0.    0.]
       [   0.    0.    0.]
       ..., 
       [   0.    0.    0.]
       [   0.    0.    0.]
       [   0.    0.    0.]]
    
      [[   1.    0.    0.]
       [   1.    0.    0.]
       [   6.    2.    1.]
       ..., 
       [  65.   94.   74.]
       [  66.   95.   73.]
       [  65.   94.   72.]]]]
    '''
    return np.asarray(X)


# X = load_img_asarray('data2', 'jpg')
# print(X)
