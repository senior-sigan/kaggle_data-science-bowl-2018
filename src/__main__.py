# -*- coding: utf-8 -*-
import sys

from data import make_train_generator
from params import Params
from unet_model import UNetModel


def build_params(argv):
    if argv[1] == 'local':
        print("local")
        return Params(train_path='../input/stage1_train/',
                      test_path='../input/stage1_test/',
                      tensorboard_dir='/tmp/tensorflow/',
                      chekpoints_path='../output/')
    elif argv[1] == 'devbox':
        print("Devbox")
        return Params(train_path='/home/ilya/Data/bowl2018/input/stage1_train/',
                      test_path='/home/ilya/Data/bowl2018/input/stage1_test/',
                      tensorboard_dir='/home/ilya/Data/bowl2018/tensorboard/',
                      chekpoints_path='/home/ilya/Data/bowl2018/output/')


def main():
    params = build_params(sys.argv)
    model = UNetModel(params)
    train_gen, valid_gen = make_train_generator(params)
    model.train(train_gen, valid_gen)


if __name__ == '__main__':
    main()
