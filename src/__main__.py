# -*- coding: utf-8 -*-

import argparse

from params import Params


def build_params(args):
    import config
    if args.local:
        print("local")
        return config.local
    elif args.devbox:
        print("Devbox")
        return config.devbox


def train(params: Params):
    from unet_model import UNetModel
    from data import make_train_generator
    model = UNetModel(params)
    train_gen, valid_gen = make_train_generator(params)
    model.train(train_gen, valid_gen)


def predict(params: Params):
    from keras.models import load_model
    from submission import make_submission
    from metrics import mean_iou, dice_coef_loss, dice_coef
    model = load_model(params.model_path,
                       {'mean_iou': mean_iou, 'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss})
    make_submission(model, params)


def main():
    parser = argparse.ArgumentParser(prog="Bowl 2018")

    env_group = parser.add_argument_group("env")
    env_group.add_argument("--local", action='store_true')
    env_group.add_argument("--devbox", action='store_true')

    parser.add_argument("--train", action='store_true')

    pred_group = parser.add_argument_group("prediction")
    pred_group.add_argument("--predict", action='store_true')
    pred_group.add_argument("model_path", type=str)

    args = parser.parse_args()
    params = build_params(args)
    if args.train:
        params.setup_train()
        train(params)
    elif args.predict:
        params.setup_submission()
        params.model_path = args.model_path
        predict(params)


if __name__ == '__main__':
    main()
