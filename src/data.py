# -*- coding: utf-8 -*-
import os
from glob import glob

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from skimage.io import imread
from sklearn.model_selection import train_test_split

from params import Params


def make_train_generator(params: Params):
    """
    Find, read and build train and validation generators
    :param params:
    :return:
    """
    print("Loading data")
    X_train, Y_train = make_train_df(params)
    # TODO: it should be numpy matrix
    X_train = np.array([read_image(x) for x in X_train])
    Y_train = np.array([flatten_masks(y) for y in Y_train])
    xtr, xval, ytr, yval = train_test_split(X_train, Y_train, test_size=params.validation_size)
    train_generator, val_generator = generator(xtr, xval, ytr, yval, params.batch_size)
    print("Data loaded")
    return train_generator, val_generator


def make_train_df(params: Params):
    """
    Load paths for train data set
    :param params:
    :return: list of pairs of image and list of masks for this image
    """
    train_root = params.train_path

    train_ids = next(os.walk(train_root))[1]
    print("Find {} train_ids".format(len(train_ids)))

    X_train = [_img_path(i, train_root) for i in train_ids]
    Y_train = [_mask_paths(i, train_root) for i in train_ids]

    return X_train, Y_train


def make_test_df(params: Params):
    test_root = params.test_path
    test_ids = next(os.walk(test_root))[1]
    print("Find {} test_ids".format(len(test_ids)))
    return [_img_path(i, test_root) for i in test_ids]


def generator(X_train, X_test, Y_train, Y_test, batch_size):
    data_gen_args = dict(horizontal_flip=True,
                         vertical_flip=True,
                         rotation_range=90.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.1)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    image_datagen.fit(X_train) # TODO: check the X_train is matrix
    mask_datagen.fit(Y_train)
    image_generator = image_datagen.flow(X_train, batch_size=batch_size, seed=7)
    mask_generator = mask_datagen.flow(Y_train, batch_size=batch_size, seed=7)
    train_generator = zip(image_generator, mask_generator)

    val_gen_args = dict()
    image_datagen_val = ImageDataGenerator(**val_gen_args)
    mask_datagen_val = ImageDataGenerator(**val_gen_args)
    image_datagen_val.fit(X_test)
    mask_datagen_val.fit(Y_test)
    image_generator_val = image_datagen_val.flow(X_test, batch_size=batch_size)
    mask_generator_val = mask_datagen_val.flow(Y_test, batch_size=batch_size)
    val_generator = zip(image_generator_val, mask_generator_val)

    return train_generator, val_generator


def _img_path(img_id, path):
    return os.path.join(path, img_id, 'images', img_id) + '.png'


def _mask_paths(img_id, path):
    masks_root = os.path.join(path, img_id, 'masks', '*') + '.png'
    return glob(masks_root)


def read_image(file_path):
    return imread(file_path)


def flatten_masks(files_paths):
    return np.sum(np.stack([read_image(file) for file in files_paths], 0), 0)
