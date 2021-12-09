
import os
import yaml
from torch.utils.data import Dataset, DataLoader
from yaml import Loader

from pdb import set_trace as st

def get_train_errors(filenames, files_nb):
    '''
    the code to get the final train_error and test_errors.
    filenames are the files you want to loop over.
    For our situation, it usually stops with '/mdl_random_mdl_{i}/meta_data.yml'
    '''
    print()
    train_errors = []
    for i in range(files_nb):

        f = filenames.replace('{}',str(i))
        with open(f, 'r') as stream:
            try:
                dict = yaml.safe_load(stream) # looks like a dictionary
                train_errors.append(dict['train_error'])
            except yaml.YAMLError as exc:
                print(exc)
    print('train_errors:')
    print(train_errors)
    return train_errors

def get_test_errors(filenames, files_nb):
    '''
    the code to get the final train_error and test_errors.
    filenames are the files you want to loop over.
    For our situation, it usually stops with '/mdl_random_mdl_{i}/meta_data.yml'
    '''
    test_errors = []
    for i in range(files_nb):
        f = filenames.replace('{}',str(i))
        with open(f, 'r') as stream:
            try:
                dict = yaml.safe_load(stream) # looks like a dictionary
                # print(dict)
                test_errors.append(dict['test_error']) # okay, now I got this number
            except yaml.YAMLError as exc:
                print(exc)
    print('test_errors:')
    print(test_errors)
    return test_errors


min_train_errors = []
min_test_errors = []

def get_min_train_errors(filenames, files_nb):
    '''
    the code to get the minimum train_error
    filenames are the files you want to loop over.
    For our situation, it usually stops with '/mdl_random_mdl_{i}/meta_data.yml'
    '''
    min_train_errors = []
    for i in range(files_nb):

        f = filenames.replace('{}',str(i))
        with open(f, 'r') as stream:
            try:
                dict = yaml.safe_load(stream)
                min_train_errors.append(min(dict['train_errors']))
            except yaml.YAMLError as exc:
                print(exc)
    print('min_train_errors:')
    print(min_train_errors)
    return min_train_errors

def get_min_test_errors(filenames, files_nb):
    '''
    the code to get the minimum train_error
    filenames are the files you want to loop over.
    For our situation, it usually stops with '/mdl_random_mdl_{i}/meta_data.yml'
    '''
    min_test_errors = []
    for i in range(files_nb):

        f = filenames.replace('{}',str(i))
        with open(f, 'r') as stream:
            try:
                dict = yaml.safe_load(stream)
                min_test_errors.append(min(dict['test_errors']))
            except yaml.YAMLError as exc:
                print(exc)
    print('min_test_errors:')
    print(min_test_errors)
    return min_test_errors

if __name__ == '__main__':
    '''
    need to put the number of folders for iterations and the pwd of this folder:
    filename's format is important. replace where need to be change as {}
    for get_train_errors(,) get_test_errors(,)
    use meta_data.yml at the end.
    can store the address_string in filenames_meta for convenience

    for get_min_train_errors(,) get_min_test_errors(,)
    use 'other_data.yml' at the end
    can store the address_string in filenames_other for convenience
    '''
    # filenames_meta = '/Users/pangda/predicting_generalization/main_full_auto_ml/data/Grp2_tower_mdl/mdl_tower_mdl_{}/meta_data.yml'
    # filenames_other = '/Users/pangda/predicting_generalization/main_full_auto_ml/data/Grp2_tower_mdl/mdl_tower_mdl_{}/other_data.yml'
    # files_nb = 27
    # get_train_errors(filenames_meta, files_nb)
    # get_test_errors(filenames_meta, files_nb)
    # get_min_train_errors(filenames_other, files_nb)
    # get_min_test_errors(filenames_other, files_nb)
