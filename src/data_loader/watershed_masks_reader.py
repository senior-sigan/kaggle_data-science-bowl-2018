# -*- coding: utf-8 -*-
import numpy as np
from scipy import ndimage
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

from src.data_loader.readers import MasksReader
from src.parallel import parallel_process


class WatershedMasksReader(MasksReader):
    def __init__(self, height, width, n_jobs=16):
        self.height = height
        self.width = width
        self.n_jobs = n_jobs

    def read(self, paths: list) -> np.ndarray:
        return np.array([img for img in self._read_resize_masks_abstract(paths)])

    def _read_resize_masks_abstract(self, files: list):
        return parallel_process(files, self._process_masks, self.n_jobs)
        # return [self._process_masks(file) for file in tqdm(files, total=len(files))]

    def _flatten_masks(self, imgs):
        return np.sum(np.stack(imgs, 0), 0)

    def _read_images(self, files_path: list):
        for file in files_path:
            yield imread(file, as_grey=True)

    def _gradient(self, imgs):
        for img in imgs:
            u, v = np.gradient(ndimage.distance_transform_edt(img))
            yield np.stack([u, v], axis=2)

    def _process_masks(self, masks: list) -> np.ndarray:
        imgs = self._read_images(masks)
        grads = self._gradient(imgs)
        grads = self._flatten_masks(grads)
        img = resize(grads, (self.height, self.width), mode='constant', preserve_range=True).astype(np.float32)
        return img
