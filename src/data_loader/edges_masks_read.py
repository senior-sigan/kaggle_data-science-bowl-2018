# -*- coding: utf-8 -*-
import numpy as np
from skimage.io import imread
from skimage.segmentation import find_boundaries
from skimage.transform import resize

from src.data_loader.readers import MasksReader
from src.parallel import parallel_process


class EdgesMasksReader(MasksReader):
    def __init__(self, height, width, n_jobs=16):
        self.height = height
        self.width = width
        self.n_jobs = n_jobs
        self._iterations = 3

    def read(self, paths: list) -> np.ndarray:
        """
        Generates image with 2 filters
        :param paths:
        :return:
        """
        return np.array([img for img in self._read_resize_masks_abstract(paths)])

    def _read_resize_masks_abstract(self, files):
        return parallel_process(files, self._process_masks, n_jobs=self.n_jobs)

    def _process_masks(self, files_paths):
        img = np.sum(np.stack([self._read_image(file) for file in files_paths], 0), 0).astype(np.bool)
        img_edged = self._remove_edges(img)
        img_edged = resize(img_edged, (self.height, self.width), mode='constant', preserve_range=True).astype(np.uint8)
        img = resize(img, (self.height, self.width), mode='constant', preserve_range=True).astype(np.uint8)
        return np.stack((img, img_edged), axis=2)

    def _read_image(self, file_path):
        return imread(file_path)[:, :]

    def _remove_edges(self, img):
        i = img
        for _ in range(self._iterations):
            b = find_boundaries(i)
            i = np.logical_not(np.logical_not(i.astype('bool')) | b)
        return i
