# -*- coding: utf-8 -*-
import os
from glob import glob

import numpy as np
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

from src.data_loader.readers import ImagesReader
from src.data_loader.simple_masks_reader import SimpleMasksReader


class WatershedImagesReader(ImagesReader):
    def __init__(self, height, width):
        self.maskReader = SimpleMasksReader(height, width)
        self.height = height
        self.width = width
        self.channels = 4

    def read(self, paths: list) -> (np.ndarray, list):
        imgs = np.zeros((len(paths), self.height, self.width, self.channels), dtype=np.uint8)
        sizes = [None] * len(paths)

        for i, file in tqdm(enumerate(paths), total=len(paths)):
            imgs[i], sizes[i] = self._process(file)

        return imgs, sizes

    def _process(self, file):
        img = self.read_image(file)
        size = (img.shape[0], img.shape[1])
        img = resize(img, (self.height, self.width), mode='constant', preserve_range=True)
        mask = self.read_mask(file)[0, :, :, :] * 255
        img = np.concatenate((img, mask), axis=2)
        return img, size

    def read_mask(self, path):
        img_dir = os.path.dirname(os.path.dirname(path))
        masks_root = os.path.join(img_dir, 'masks', '*') + '.png'
        masks = glob(masks_root)
        return self.maskReader.read([masks])

    def read_image(self, file_path):
        return imread(file_path)[:, :, :self.channels - 1]
