from os.path import isfile
from scipy.io import loadmat, savemat
import numpy as np

### Constants
GENDER_FEMALE = 0
GENDER_MALE = 1
VERBOSE = False


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
    name_list, full_path_list, gender_list, face_score_list, face_location_list, dob_list, photo_taken_list, second_face_score_list = load_label(data_dir + '/wiki.mat')

    female_full_paths = []
    female_genders = []
    male_full_paths = []
    male_genders = []
    for name, full_path, gender, face_score, second_face_score in zip(name_list, full_path_list, gender_list, face_location_list, second_face_score_list):
        ### Ignore bad data
        if gender != GENDER_FEMALE and gender != GENDER_MALE:
            if VERBOSE:
                print('Pass gender NaN:', full_path[0])
            continue
        if not np.isnan(second_face_score):
            if VERBOSE:
                print('Pass second face score not NaN:', full_path[0])
            continue
        if not isfile(data_dir + '/' + full_path[0].strip()):
            if VERBOSE:
                print('Pass not exist:', full_path[0])
            continue
        # if face_score > 1.0:
        #     continue

        if gender == GENDER_FEMALE:
            female_full_paths.append( full_path[0] )
            female_genders.append( gender )
        if gender == GENDER_MALE:
            male_full_paths.append( full_path[0] )
            male_genders.append( gender )

    savemat('female_wiki.mat', { "full_path": np.array(female_full_paths), "gender": np.array(female_genders) })
    savemat('male_wiki.mat',   { "full_path": np.array(  male_full_paths), "gender": np.array(  male_genders) })
    print('Data saved:')
    print('   Female:', len(female_genders))
    print('   Male  :', len(male_genders))


### Modify wiki.mat and create a modified mat file. TODO: Change the function name
load_img_label_asarray('data', 'jpg')
