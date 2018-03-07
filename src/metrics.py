# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from keras import backend as K


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)


def cos_loss(y_true, y_pred):
    # TODO: it's not working
    norma = tf.linalg.norm(y_true) * tf.linalg.norm(y_pred)
    theta_ = tf.tensordot(y_true, y_pred, axes=3) / norma
    theta = K.clip(theta_, -1.0, 1.0)
    return tf.reshape(tf.acos(theta), [])


def angle_loss(y_true, y_pred):
    y_pred = tf.nn.l2_normalize(y_pred, 1) * 0.999999
    y_true = tf.nn.l2_normalize(y_true, 1) * 0.999999
    error_angles = tf.acos(tf.reduce_sum(y_pred * y_true, reduction_indices=[1], keep_dims=True))
    return tf.reduce_sum((tf.abs(error_angles * error_angles)))
