from six import moves
import tensorflow as tf

from ..math import matmul_l_2
from ..shaping import mixed_shape


def my_sum(l):
    if len(l) == 0:
        return 0
    elif len(l) == 1:
        return l[0]
    else:
        r = 0
        for i in moves.xrange(len(l)):
            r += l[i]
        return r


def linear(x, hid_dim, use_bias=True,
           bias_initializer=tf.zeros_initializer(),
           weight_initializer=tf.glorot_uniform_initializer(),
           weight_regularizer=None,
           scope=None, reuse=None):
    """
    x is a list/tuple of tensors.
    Do NOT perform expensive concat.
    Just compute the linear transformation for all tensor in x
    """
    if isinstance(x, (list, tuple)):
        x = list(x)
    else:
        x = [x]

    x_dims = [mixed_shape(x_)[-1] for x_ in x]
    assert all([isinstance(x_dim, int) for x_dim in x_dims])

    with tf.variable_scope(scope or "linear", reuse=reuse):
        if len(x_dims) == 1:
            # W_names = ["W"]
            w_names = ["w"]
        else:
            # W_names = ["W_{}".format(i) for i in range(len(x_dims))]
            w_names = ["w_{}".format(i) for i in range(len(x_dims))]

        w = [tf.get_variable(w_names[i], shape=[x_dim, hid_dim], dtype=tf.float32,
                             initializer=weight_initializer, regularizer=weight_regularizer)
             for i, x_dim in enumerate(x_dims)]

        h = my_sum([matmul_l_2(x_, w_) for x_, w_ in zip(x, w)])
        if use_bias:
            b = tf.get_variable("b", shape=[hid_dim], dtype=tf.float32,
                                initializer=bias_initializer)
            h += b
        return h


def fc(x, hid_dim, activation=None,
       use_bias=True, bias_initializer=tf.zeros_initializer(),
       weight_initializer=tf.glorot_uniform_initializer(),
       weight_regularizer=None,
       scope=None, reuse=None):

    if isinstance(x, (list, tuple)):
        x = list(x)
    else:
        x = [x]

    x_dims = [mixed_shape(x_)[-1] for x_ in x]
    assert all([isinstance(x_dim, int) for x_dim in x_dims])

    with tf.variable_scope(scope or "fc", reuse=reuse):
        if len(x_dims) == 1:
            w_names = ["w"]
        else:
            w_names = ["w_{}".format(i) for i in range(len(x_dims))]

        w = [tf.get_variable(w_names[i], shape=[x_dim, hid_dim], dtype=tf.float32,
                             initializer=weight_initializer, regularizer=weight_regularizer)
             for i, x_dim in enumerate(x_dims)]

        y = my_sum([matmul_l_2(x_, w_) for x_, w_ in zip(x, w)])

        if use_bias:
            b = tf.get_variable("b", shape=[hid_dim], dtype=tf.float32,
                                initializer=bias_initializer)
            y += b

        if activation is not None:
            y = activation(y)

        return y