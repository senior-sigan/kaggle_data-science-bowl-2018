# -*- coding: utf-8 -*-
import sys

from data import make_train_generator
import config
from unet_model import UNetModel


def build_params(argv):
    if len(argv) == 1 or argv[1] == 'local':
        print("local")
        return config.local
    elif argv[1] == 'devbox':
        print("Devbox")
        return config.devbox


def main():
    params = build_params(sys.argv)
    model = UNetModel(params)
    train_gen, valid_gen = make_train_generator(params)
    model.train(train_gen, valid_gen)


if __name__ == '__main__':
    main()
