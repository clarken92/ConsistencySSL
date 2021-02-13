from __future__ import absolute_import

import tensorflow as tf


def activation_from_str(s):
    str2act = {
        'sigmoid': tf.nn.sigmoid,
        'relu': tf.nn.relu,
        'leaky_relu': tf.nn.leaky_relu,
        'selu': tf.nn.selu,
    }
    return str2act[s]


def exp_w_clip(x, min=-50.0, max=50.0, name=None):
    with tf.name_scope(name or "exp_w_clip"):
        return tf.exp(tf.clip_by_value(x, min, max))


def log_w_clip(x, min=1e-15, max=1e30, name=None):
    with tf.name_scope(name or "log_w_clip"):
        return tf.log(tf.clip_by_value(x, min, max))


def sqrt_w_clip(x, min=1e-15, max=1e15, name=None):
    with tf.name_scope(name or "sqrt_w_clip"):
        return tf.sqrt(tf.clip_by_value(x, min, max))


# We take log then we take exp
def sqrt_w_clip_v2(x, min=1e-15, max=1e15, name=None):
    with tf.name_scope(name or "sqrt_w_clip"):
        return exp_w_clip(log_w_clip(x, min=min, max=max) * 0.5)
