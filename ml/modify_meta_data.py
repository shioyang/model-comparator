from os.path import isfile
from scipy.io import loadmat, savemat
import numpy as np

###
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

    full_paths = []
    genders = []
    for name, full_path, gender, face_score, second_face_score in zip(name_list, full_path_list, gender_list, face_location_list, second_face_score_list):
        ### Ignore bad data
        if gender != 0 and gender != 1:
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

        full_paths.append( full_path[0] )
        genders.append( gender )

    ### Count female and male
    countFemale = 0
    countMale = 0
    for gender in genders:
        if gender == 0:
            countFemale += 1
        if gender == 1:
            countMale += 1
    print('Data origin:')
    print('   Female:', countFemale)
    print('   Male  :', countMale)
    count = min([countFemale, countMale])

    countImgFemale = 0
    countImgMale = 0
    new_full_paths = []
    new_genders = []
    for full_path, gender in zip(full_paths, genders):
        if gender == 0:
            if countImgFemale >= count:
                continue
            countImgFemale += 1
        if gender == 1:
            if countImgMale >= count:
                continue
            countImgMale += 1
        new_full_paths.append(full_path)
        new_genders.append(gender)

    savemat('modified_wiki.mat', { "full_path": np.array(new_full_paths), "gender": np.array(new_genders) })
    print('Data saved:')
    print('   Female:', countImgFemale)
    print('   Male  :', countImgMale)


### Modify wiki.mat and create a modified mat file. TODO: Change the function name
load_img_label_asarray('data', 'jpg')
