# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
from keras.engine import Model
from skimage.transform import resize
from tqdm import tqdm

from data import make_test_df, read_resize_images
from params import Params
from rle_encodign import prob_to_rles


def make_submission(model: Model, params: Params):
    tests_paths = make_test_df(params)
    test_imgs, sizes = read_resize_images(tests_paths)
    assert test_imgs.ndim == 4
    masks = model.predict(test_imgs)

    data = []
    for i, path in tqdm(enumerate(tests_paths), total=len(tests_paths), desc='Building Rles'):
        mask = masks[i]
        size = sizes[i]
        name = os.path.basename(path)[:-4]
        use_edges = False  # TODO it was test to use edges, seems to be terrible
        if use_edges:
            edges = mask[:, :, 1].reshape((256, 256)) < params.cutoff
            segments = mask[:, :, 0].reshape((256, 256)) > params.cutoff
            nucleis = segments & edges  # remove edges so we decouple close nucleis
            result = resize(nucleis, size, mode='constant', preserve_range=True)
        else:
            segments = mask[:, :].reshape((256, 256)) > params.cutoff
            result = resize(segments, size, mode='constant', preserve_range=True)

        rles = list(prob_to_rles(result))
        for rle in rles:
            data.append({
                'ImageId': name,
                'EncodedPixels': ' '.join(np.array(rle).astype(str))
            })

    df = pd.DataFrame(data)
    df[['ImageId', 'EncodedPixels']].to_csv(params.submission_path, index=False)
    return df
