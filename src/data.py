# -*- coding: utf-8 -*-
import os
from glob import glob

import numpy as np
from sklearn.model_selection import train_test_split

from src.data_loader.readers import MasksReader, ImagesReader
from src.params import Params


def make_train_generator(params: Params):
    from src.data_loader.simple_images_reader import SimpleImagesReader
    from src.data_loader.simple_masks_reader import SimpleMasksReader

    return _make_train_generator(params, SimpleImagesReader(256, 256), SimpleMasksReader(256, 256))


def make_watershed_train_generator(params: Params):
    from src.data_loader.watershed_image_reader import WatershedImagesReader
    from src.data_loader.watershed_masks_reader import WatershedMasksReader
    return _make_train_generator(params, WatershedImagesReader(256, 256), WatershedMasksReader(256, 256, params.n_jobs))


def _make_train_generator(params: Params, images_reader: ImagesReader, masks_reader: MasksReader):
    """
    Find, read and build train and validation generators
    :param params:
    :return:
    """
    print("Loading data")
    X_train, Y_train = make_train_df(params)
    xtr, xval, ytr, yval = train_test_split(X_train, Y_train, test_size=params.validation_size)
    train_generator, val_generator = generator(xtr, xval, ytr, yval, params.batch_size, images_reader, masks_reader)
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

    if params.sample:
        return X_train[:params.sample], Y_train[:params.sample]
    else:
        return X_train, Y_train


def make_test_df(params: Params):
    test_root = params.test_path
    test_ids = next(os.walk(test_root))[1]
    print("Find {} test_ids".format(len(test_ids)))
    return [_img_path(i, test_root) for i in test_ids]


def generator(X_train, X_test, Y_train, Y_test, batch_size,
              images_reader: ImagesReader, masks_reader: MasksReader):
    from keras.preprocessing.image import ImageDataGenerator
    seed = 42

    Y_train = masks_reader.read(Y_train)
    Y_test = masks_reader.read(Y_test)
    X_train, _ = images_reader.read(X_train)
    X_test, _ = images_reader.read(X_test)

    data_gen_args = dict(horizontal_flip=True,
                         vertical_flip=True,
                         rotation_range=90.,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         zoom_range=0.4,
                         fill_mode='reflect')
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    image_datagen.fit(X_train, seed=seed)
    mask_datagen.fit(Y_train, seed=seed)
    image_generator = image_datagen.flow(X_train, batch_size=batch_size, seed=seed)
    mask_generator = mask_datagen.flow(Y_train, batch_size=batch_size, seed=seed)
    train_generator = zip(image_generator, mask_generator)

    val_gen_args = dict()
    image_datagen_val = ImageDataGenerator(**val_gen_args)
    mask_datagen_val = ImageDataGenerator(**val_gen_args)
    image_datagen_val.fit(X_test, seed=seed)
    mask_datagen_val.fit(Y_test, seed=seed)
    image_generator_val = image_datagen_val.flow(X_test, batch_size=batch_size, seed=seed)
    mask_generator_val = mask_datagen_val.flow(Y_test, batch_size=batch_size, seed=seed)
    val_generator = zip(image_generator_val, mask_generator_val)

    return train_generator, val_generator


def _img_path(img_id, path):
    return os.path.join(path, img_id, 'images', img_id) + '.png'


def _mask_paths(img_id, path):
    masks_root = os.path.join(path, img_id, 'masks', '*') + '.png'
    return np.array(glob(masks_root))
