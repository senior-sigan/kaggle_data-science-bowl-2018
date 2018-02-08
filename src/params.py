# -*- coding: utf-8 -*-

from datetime import datetime


class Params:
    def __init__(self, train_path, test_path, tensorboard_dir, chekpoints_path, sample=None):
        t = int(datetime.now().timestamp())
        self.train_path = train_path
        self.test_path = test_path
        self.tensorboard_dir = "{}--{}".format(tensorboard_dir, t)
        self.chekpoints_path = "{}--{}".format(chekpoints_path, t)
        self.validation_size = 0.2
        self.batch_size = 32
        self.epochs = 30
        self.validation_steps_per_epoch = 670 * self.validation_size
        self.steps_per_epoch = 670 * (1 - self.validation_size)
        self.sample = sample
