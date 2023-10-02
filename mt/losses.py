"""
Losses uses train models for MT tracing
"""

import tensorflow as tf


def l2_norm(y_true, y_pred):
    """
    L2 Norm, the squared error
    :param y_true: true values vector
    :param y_pred: predicted values vector
    :return: loss a an scalar
    """
    return tf.reduce_sum((y_true - y_pred)**2)