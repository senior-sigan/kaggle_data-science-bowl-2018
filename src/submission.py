# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
from keras.engine import Model
from skimage.transform import resize
from tqdm import tqdm

from data import make_test_df, read_resize_images, make_train_df
from params import Params
from rle_encodign import prob_to_rles


def make_submission(model: Model, params: Params):
    tests_paths = make_test_df(params)
    test_imgs, sizes = read_resize_images(tests_paths)
    assert test_imgs.ndim == 4
    masks = model.predict(test_imgs)

    df = calc_rles(tests_paths, masks, sizes, params)
    df[['ImageId', 'EncodedPixels']].to_csv(params.submission_path, index=False)
    return df


def calc_rles(file_paths: list, masks: list, sizes: list, params: Params):
    data = []
    for i, path in tqdm(enumerate(file_paths), total=len(file_paths), desc='Building Rles'):
        mask = masks[i]
        size = sizes[i]
        name = os.path.basename(path)[:-4]
        segments = mask[:, :].reshape((256, 256)) > params.cutoff
        result = resize(segments, size, mode='constant', preserve_range=True)

        rles = list(prob_to_rles(result))
        for rle in rles:
            data.append({
                'ImageId': name,
                'EncodedPixels': ' '.join(np.array(rle).astype(str))
            })

    return pd.DataFrame(data)


def train_score(model: Model, params: Params):
    X_train_paths, _ = make_train_df(params)
    X_train, sizes = read_resize_images(X_train_paths)
    assert X_train.ndim == 4
    masks = model.predict(X_train, verbose=1)

    predicted = calc_rles(X_train_paths, masks, sizes, params)
    origin = pd.read_csv(params.train_rles_path)
    score = submission_score(predicted, origin)
    print("Train score: {}".format(score))


def submission_score(predicted_labels: pd.DataFrame, train_labels: pd.DataFrame) -> float:
    pass
