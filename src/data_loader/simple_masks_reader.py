# -*- coding: utf-8 -*-
import numpy as np
from skimage.io import imread
from skimage.transform import resize

from data_loader.readers import MasksReader
from parallel import parallel_process


class SimpleMasksReader(MasksReader):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def read(self, paths: list) -> np.ndarray:
        return np.array([img for img in self._read_resize_masks_abstract(paths)])

    def _read_resize_masks_abstract(self, files):
        for img in parallel_process(files, self._process_masks):
            yield resize(img, (self.height, self.width), mode='constant', preserve_range=True).astype(np.uint8)

    def _process_masks(self, files_paths):
        return np.sum(np.stack([self._read_image(file) for file in files_paths], 0), 0).astype(np.bool)

    def _read_image(self, file_path):
        return np.expand_dims(imread(file_path)[:, :], axis=-1)
