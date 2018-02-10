# -*- coding: utf-8 -*-
from params import Params

local = Params(train_path='/Users/ilya/Documents/machine_learning/kaggle_data-science-bowl-2018/input/stage1_train/',
               test_path='/Users/ilya/Documents/machine_learning/kaggle_data-science-bowl-2018/input/stage1_test/',
               train_rles_path='/Users/ilya/Documents/machine_learning/kaggle_data-science-bowl-2018/input/stage1_train_labels.csv',
               tensorboard_dir='/tmp/tensorflow/',
               chekpoints_path='/Users/ilya/Documents/machine_learning/kaggle_data-science-bowl-2018/output/',
               submission_dir='/Users/ilya/Documents/machine_learning/kaggle_data-science-bowl-2018/submissions/',
               sample=5)

devbox = Params(train_path='/home/ilya/Data/bowl2018/input/stage1_train/',
                test_path='/home/ilya/Data/bowl2018/input/stage1_test/',
                train_rles_path='/home/ilya/Data/bowl2018/input/stage1_train_labels.csv',
                tensorboard_dir='/home/ilya/Data/bowl2018/tensorboard/',
                chekpoints_path='/home/ilya/Data/bowl2018/output/',
                submission_dir='/home/ilya/Data/bowl2018/submissions/',
                sample=None)
