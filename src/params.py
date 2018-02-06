# -*- coding: utf-8 -*-


class Params:
    def __init__(self, train_path, test_path, tensorboard_dir, chekpoints_path):
        self.train_path = train_path
        self.test_path = test_path
        self.tensorboard_dir = tensorboard_dir
        self.chekpoints_path = chekpoints_path
        self.validation_size = 0.1
        self.batch_size = 32
        self.epochs = 30
        self.validation_steps_per_epoch = 670 * self.validation_size
        self.steps_per_epoch = 670 * (1 - self.validation_size)
