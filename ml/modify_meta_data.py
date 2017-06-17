from os.path import isfile
from scipy.io import loadmat, savemat
import numpy as np


def load_label(data_path_file):
    meta = loadmat(data_path_file)

    name              = meta['wiki'][0, 0]['name'][0]
    full_path         = meta['wiki'][0, 0]['full_path'][0]
    gender            = meta['wiki'][0, 0]['gender'][0]
    face_score        = meta['wiki'][0, 0]['face_score'][0]
    face_location     = meta['wiki'][0, 0]['face_location'][0]
    dob               = meta['wiki'][0, 0]['dob'][0]               # Day of birth
    photo_taken       = meta['wiki'][0, 0]['photo_taken'][0]
    second_face_score = meta['wiki'][0, 0]['second_face_score'][0]

    return name, full_path, gender, face_score, face_location, dob, photo_taken, second_face_score


def load_img_label_asarray(data_dir, ext):
    full_paths = []
    genders = []

    name_list, full_path_list, gender_list, face_score_list, face_location_list, dob_list, photo_taken_list, second_face_score_list = load_label(data_dir + '/wiki.mat')

    for name, full_path, gender, face_score, second_face_score in zip(name_list, full_path_list, gender_list, face_location_list, second_face_score_list):
        if gender != 0 and gender != 1:
            print('Pass gender NaN:', full_path[0])
            continue
        if not np.isnan(second_face_score):
            print('Pass second face score not NaN:', full_path[0])
            continue
        if not isfile(data_dir + '/' + full_path[0].strip()):
            print('Pass not exist:', full_path[0])
            continue
        # if face_score > 1.0:
        #     continue
        full_paths.append( full_path[0] )
        genders.append( gender )

    savemat('modified_wiki.mat', { "full_path": np.array(full_paths), "gender": np.array(genders) })


### Modify wiki.mat and create a modified mat file. TODO: Change the function name
load_img_label_asarray('data', 'jpg')
