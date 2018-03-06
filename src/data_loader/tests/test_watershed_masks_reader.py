# -*- coding: utf-8 -*-

import unittest
from glob import glob

from data_loader.watershed_masks_reader import WatershedMasksReader


class TestWatershedMasksReader(unittest.TestCase):
    def test_reader(self):
        path = "/Users/ilya/Documents/machine_learning/kaggle_data-science-bowl-2018/input/stage1_train/ff599c7301daa1f783924ac8cbe3ce7b42878f15a39c2d19659189951f540f48/masks/*.png"
        reader = WatershedMasksReader(256, 256)
        mask = reader.read([(glob(path))])
        self.assertEqual((1, 256, 256, 2), mask.shape)


if __name__ == '__main__':
    unittest.main()
