# -*- coding: utf-8 -*-
import numpy as np
from scipy import ndimage
from skimage.io import imread
from skimage.transform import resize

from data_loader.readers import MasksReader
from parallel import parallel_process


class WatershedMasksReader(MasksReader):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def read(self, paths: list) -> np.ndarray:
        return np.array([img for img in self._read_resize_masks_abstract(paths)])

    def _read_resize_masks_abstract(self, files: list):
        for img in parallel_process(files, self._process_masks):
            yield resize(img, (self.height, self.width), mode='constant', preserve_range=True).astype(np.uint8)

    def _flatten_masks(self, imgs):
        return np.sum(np.stack(imgs, 0), 0)

    def _read_images(self, files_path: list):
        for file in files_path:
            yield imread(file)

    def _gradient(self, imgs):
        for img in imgs:
            yield np.gradient(ndimage.distance_transform_edt(img))

    def _angle(self, gradients, imgs):
        for grad, img in zip(gradients, imgs):
            yield self._to_angle(grad[1], grad[0], img)

    def _to_angle(self, v, u, mask):
        angle = np.angle(v + u * 1.0j, deg=True)
        m = np.logical_or(angle, mask > 0)
        angle = angle + m * 360.0  # подтягиваем наверх градусы, чтобы 0 - это был фон, а не 0 угол
        return (angle / 540.0) * 255.0

    def _normalize(self, angle):
        angle[angle > 255] = 255
        angle[angle < 0] = 0
        return angle

    def _process_masks(self, masks: list) -> np.ndarray:
        imgs = list(self._read_images(masks))
        grads = self._gradient(imgs)
        angles = self._angle(grads, imgs)
        angle = self._normalize(self._flatten_masks(angles))

        return np.expand_dims(angle, axis=-1)
