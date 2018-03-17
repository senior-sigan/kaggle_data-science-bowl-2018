# -*- coding: utf-8 -*-

import os
from datetime import datetime


class Params:
    def __init__(self, train_path, test_path, train_rles_path, tensorboard_dir, chekpoints_path, submission_dir,
                 n_jobs=16,
                 sample=None):
        self.model_path = None
        self.train_path = train_path
        self.test_path = test_path
        self.validation_size = 0.1
        self.batch_size = 32
        self.epochs = 100
        steps = 670
        self.validation_steps_per_epoch = steps * self.validation_size
        self.steps_per_epoch = steps * (1 - self.validation_size)
        self.sample = sample
        self.cutoff = 0.5
        self.submission_dir = submission_dir
        self.submission_path = None
        self.train_rles_path = train_rles_path
        self.tensorboard_dir = tensorboard_dir
        self.chekpoints_path = chekpoints_path
        self.n_jobs = n_jobs

    def setup_train(self, name):
        t = int(datetime.now().timestamp())
        self.tensorboard_dir = os.path.join(self.tensorboard_dir, "{}--{}".format(name, t))
        self.chekpoints_path = os.path.join(self.chekpoints_path, "{}--{}".format(name, t))
        print(self.tensorboard_dir)
        os.makedirs(self.tensorboard_dir)
        print(self.chekpoints_path)
        os.makedirs(self.chekpoints_path)

    def setup_submission(self):
        t = int(datetime.now().timestamp())
        print(self.submission_dir)
        os.makedirs(self.submission_dir, exist_ok=True)
        self.submission_path = os.path.join(self.submission_dir, "submission--{}.csv".format(t))
        return self
