# -*- coding: utf-8 -*-

import os
from datetime import datetime


class Params:
    def __init__(self, train_path, test_path, tensorboard_dir, chekpoints_path, sample=None):
        t = int(datetime.now().timestamp())
        name = 'unet'
        self.train_path = train_path
        self.test_path = test_path
        self.tensorboard_dir = os.path.join(tensorboard_dir, "{}--{}".format(name, t))
        self.chekpoints_path = os.path.join(chekpoints_path, "{}--{}".format(name, t))
        self.validation_size = 0.2
        self.batch_size = 32
        self.epochs = 30
        self.validation_steps_per_epoch = 670 * self.validation_size
        self.steps_per_epoch = 670 * (1 - self.validation_size)
        self.sample = sample

    def setup(self):
        print(self.tensorboard_dir)
        os.makedirs(self.tensorboard_dir)
        print(self.chekpoints_path)
        os.makedirs(self.chekpoints_path)
        return self