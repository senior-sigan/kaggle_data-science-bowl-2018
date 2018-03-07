# -*- coding: utf-8 -*-

import os
import unittest
from glob import glob

import numpy as np

from src.config import devbox as conf
from src.data_loader.watershed_masks_reader import WatershedMasksReader


class WatershedMasksReaderTest(unittest.TestCase):
    def test_reader(self):
        mask_path = "ff599c7301daa1f783924ac8cbe3ce7b42878f15a39c2d19659189951f540f48/masks/*.png"
        path = os.path.join(conf.train_path, mask_path)
        reader = WatershedMasksReader(256, 256)
        print(path)
        mask = reader.read([(glob(path)), (glob(path)), (glob(path))])
        self.assertEqual((3, 256, 256, 2), mask.shape)

    def test_read_all(self):
        reader = WatershedMasksReader(256, 256, n_jobs=conf.n_jobs)
        train_ids = next(os.walk(conf.train_path))[1]
        masks = reader.read([_mask_paths(i, conf.train_path) for i in train_ids])
        print(masks.shape)
        self.assertEqual(masks.shape, (len(train_ids), 256, 256, 2))


def _mask_paths(img_id, path):
    masks_root = os.path.join(path, img_id, 'masks', '*') + '.png'
    return np.array(glob(masks_root))


if __name__ == '__main__':
    unittest.main()
