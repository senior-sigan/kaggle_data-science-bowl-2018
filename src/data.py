# -*- coding: utf-8 -*-
import os
from glob import glob
from random import sample

import numpy as np
from scipy import ndimage
from skimage import feature
from skimage.io import imread, imsave
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from parallel import parallel_process
from params import Params


def make_train_generator(params: Params):
    """
    Find, read and build train and validation generators
    :param params:
    :return:
    """
    print("Loading data")
    X_train, Y_train = make_train_df(params)
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

    if params.sample:
        return sample(X_train, params.sample), sample(Y_train, params.sample)
    else:
        return X_train, Y_train


def make_test_df(params: Params):
    test_root = params.test_path
    test_ids = next(os.walk(test_root))[1]
    print("Find {} test_ids".format(len(test_ids)))
    return [_img_path(i, test_root) for i in test_ids]


def generator(X_train, X_test, Y_train, Y_test, batch_size):
    from keras.preprocessing.image import ImageDataGenerator
    X_train, _ = read_resize_images(X_train)
    X_test, _ = read_resize_images(X_test)
    Y_train = read_resize_masks(Y_train)
    Y_test = read_resize_masks(Y_test)

    data_gen_args = dict(horizontal_flip=True,
                         vertical_flip=True,
                         rotation_range=90.,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         zoom_range=0.2,
                         fill_mode='reflect')
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    image_datagen.fit(X_train)
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
    return np.array(glob(masks_root))


def read_image(file_path, channels):
    if channels != 1:
        return imread(file_path)[:, :, :channels]
    else:
        return np.expand_dims(imread(file_path)[:, :], axis=-1)


def flatten_masks(files_paths):
    return np.sum(np.stack([read_image(file, 1) for file in files_paths], 0), 0).astype(np.bool)


def flatten_mask_without_edges(files_paths):
    masks = []
    for file in files_paths:
        im = imread(file).astype(np.bool)
        edge = feature.canny(imread(file), sigma=1).astype(np.bool)
        img = (im & (edge != True))
        masks.append(np.expand_dims(img, axis=-1))

    return np.sum(np.stack(masks, 0), 0).astype(np.bool)


def flatten_masks_edges(files_paths):
    """
    Generates a mask of ceil edges
    :param files_paths:
    :return:
    """
    edges = np.sum(np.stack([feature.canny(imread(file), sigma=1) for file in files_paths], 0), 0).astype(np.bool)
    return np.expand_dims(edges, axis=-1)


def read_resize_images(files, height=256, width=256) -> (np.ndarray, list):
    imgs = np.zeros((len(files), height, width, 3), dtype=np.uint8)
    sizes = []
    for i, file in tqdm(enumerate(files), total=len(files)):
        img = read_image(file, 3)
        sizes.append((img.shape[0], img.shape[1]))
        imgs[i] = resize(img, (height, width), mode='constant', preserve_range=True)
    return imgs, sizes


def read_resize_masks(files, height=256, width=256):
    return np.array([img for img, _ in read_resize_masks_abstract(files, flatten_masks, height, width)])


def read_resize_masks_abstract(files, func, height=256, width=256):
    for img, name in parallel_process(files, func):
        yield resize(img, (height, width), mode='constant', preserve_range=True).astype(np.uint8), name


def to_angle(v, u, mask):
    angle = np.angle(v + u * 1.0j, deg=True)
    m = np.logical_or(angle, mask > 0)
    angle = angle + m * 360.0  # подтягиваем наверх градусы, чтобы 0 - это был фон, а не 0 угол
    return (angle / 540.0) * 255.0


def flatten_masks_grad(masks) -> (np.ndarray, str):
    name = os.path.basename(os.path.dirname(os.path.dirname(masks[0])))
    imgs = [imread(mask) for mask in masks]
    grads = [np.gradient(ndimage.distance_transform_edt(img)) for img in imgs]
    angles = [to_angle(grad[1], grad[0], img) for grad, img in zip(grads, imgs)]
    angle = np.sum(np.stack(angles, 0), 0)
    angle[angle > 255] = 255
    angle[angle < 0] = 0
    return np.expand_dims(angle, axis=-1), name


def save_depth_map(params):
    print("Loading data")
    X_train, Y_train = make_train_df(params)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(X_train[0])))
    print("Base dir: {}".format(base_dir))
    for mask, name in parallel_process(Y_train, flatten_masks_grad):
        dir_path = os.path.join(base_dir, name, 'depth_mask')
        os.makedirs(dir_path, exist_ok=True)
        mask_path = os.path.join(dir_path, name + '.png')

        img = mask[:, :, 0].astype(dtype=np.uint8)
        imsave(mask_path, img)
        print("{} saved".format(mask_path))
