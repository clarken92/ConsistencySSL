from __future__ import absolute_import
import tensorflow as tf
from functools import reduce
from operator import mul

from .shaping import ndims, mixed_shape, flatten_left_to, reconstruct_left


def matmul_l_2(x, y, name=None):
    """
    Matmul between 2 tensors 'x' and 'y'
    :param x: A k-dimensional tensor of shape (b1, b2,..., bk-1, m)
    :param y: A 2D matrix of shape (m, n)
    :param name: Name scope for this operator
    :return: A k-dimensional tensor of shape (b1, b2,...,bk-1, n)
    """

    assert y.get_shape().ndims == 2, "`y` must be a 2D matrix!"
    with tf.name_scope(name or "matmul_l_2"):
        if x.get_shape().ndims == 2:
            output = tf.matmul(x, y)
        else:
            x_flat = flatten_left_to(x, 1)
            output = tf.matmul(x_flat, y)
            output = reconstruct_left(output, x, 1)
        return output


def matmul_l_1(x, y, name=None):
    """
    Matmul between 2 tensors 'x' and 'y'
    :param x: A k-dimensional tensor of shape (b1, b2,..., bk-1, m)
    :param y: A 1D matrix of shape (m)
    :param name: Name scope for this operator
    :return: A k-dimensional tensor of shape (b1, b2,...,bk-1)
    """
    assert y.get_shape().ndims == 1, "`y` must be a 1D matrix!"
    with tf.name_scope(name or "matmul_l_1"):
        x_shape = mixed_shape(x)
        x_flat = tf.reshape(x, [reduce(mul, x_shape[:-1]), x_shape[-1]])
        # (b1 * b2 *...* bk-1, 1)
        output = tf.matmul(x_flat, tf.expand_dims(y, axis=1))

        assert ndims(output) == 2 and output.get_shape()[-1] == 1
        output = tf.reshape(output, x_shape[:-1])
        return output


def matmul_l_lm1(x, y, name=None):
    """
    Matmul between 2 tensors 'x' and 'y'
    :param x: A k dimensional tensor of shape (b1, b2,..., bk-2, m, n)
    :param y: A k-1 dimensional matrix of shape (b1, b2,...,bk-2, n)
    :param name: Name scope for this operator
    :return: A k-1 dimensional tensor of shape (b1, b2,...,bk-2, m)
    """
    assert ndims(x) == ndims(y) + 1, "`x` must have 1 more dimension than `y`!"
    with tf.name_scope(name or "matmul_l_lm1"):
        _y = tf.expand_dims(y, axis=ndims(y))
        out = tf.matmul(x, _y)

        assert (ndims(out) == ndims(x)) and (out.get_shape()[-1] == 1)
        out = tf.reshape(out, mixed_shape(out)[:-1])
        return out


def matmul_l_l(x, y, name=None):
    """
    Matmul between 2 tensors 'x' and 'y'
    :param x: A k dimensional tensor of shape (b1, b2,..., bk-1, n)
    :param y: A k dimensional matrix of shape (b1, b2,...,bk-1, n)
    :param name:
    :return: A k-1 dimensional tensor of shape (b1, b2,...,bk-2, bk-1)
    """
    assert x.get_shape().ndims == y.get_shape().ndims, "`x` and `y` must have the same dimensions!"
    with tf.name_scope(name or "matmul_l_l"):
        ndims = x.get_shape().ndims
        _x = tf.expand_dims(x, axis=ndims-1)
        _y = tf.expand_dims(y, axis=ndims)

        # (b1, b2,..., bk-1, 1, 1)
        out = tf.matmul(_x, _y)
        out_shape = out.get_shape()
        assert (out_shape.ndims == ndims + 2) and (out_shape[-1] == out_shape[-2] == 1)
        return tf.reshape(out, mixed_shape(out)[:-2])


def matmul_lm1_lm1(x, y, name=None):
    """
    Matmul between 2 tensors 'x' and 'y'
    :param x: A k dimensional tensor of shape (b1, b2,...,bk-2, n, p)
    :param y: A k dimensional tensor of shape (b1, b2,...,bk-2, n, q)
    :param name:
    :return: A k dimensional tensor of shape (b1, b2,...,bk-2, p, q)
    """
    assert x.get_shape().ndims == y.get_shape().ndims, "`x` and `y` must have the same dimensions!"
    assert x.get_shape().ndims >= 2
    with tf.name_scope(name or "matmul_lm1_lm1"):
        ndims = x.get_shape().ndims
        perm = [i for i in range(ndims)]
        perm[-1] = ndims-2
        perm[-2] = ndims-1

        x_ = tf.transpose(x, perm)
        out = tf.matmul(x_, y)
        return out


def batch_tensor_dot_l(x, y, num_batch_axes):
    """
    Perform tensor dot on a single axis with batch support
    x and y have the same k batch (leftmost) axes
    x and y also have the same last (rightmost) axis for summing during dot
    The middle axes of x and y are arbitrary (in number and value) and will be flatten out
    :param x: (b1, b2,..., bk, r1, r2,...,rp, n)
    :param y: (b1, b2,..., bk, s1, s2,...,sq, n)
    :param num_batch_axes: Number of batch (leftmost) axes for x and y
    :return: (b1, b2,...,bk, r1, r2,...,rp, s1, s2,...,sq)
    """

    x_shape = mixed_shape(x)
    y_shape = mixed_shape(y)
    # print("type(x_shape): {}".format(type(x_shape)))
    # print("type(y_shape): {}".format(type(y_shape)))
    assert len(x_shape) > num_batch_axes and len(y_shape) > num_batch_axes

    # Use for reconstruction
    x_middle_axes = x_shape[num_batch_axes: len(x_shape)-1]
    y_middle_axes = y_shape[num_batch_axes: len(y_shape)-1]
    if len(x_middle_axes) == 0:
        x_middle_axes = [1]
    if len(y_middle_axes) == 0:
        y_middle_axes = [1]
    # print("x_middle_axes: {}".format(x_middle_axes))
    # print("y_middle_axes: {}".format(y_middle_axes))

    # New x, y shape
    new_x_shape = x_shape[0:num_batch_axes] + [reduce(mul, x_middle_axes)] + [x_shape[len(x_shape)-1]]
    new_y_shape = y_shape[0:num_batch_axes] + [reduce(mul, y_middle_axes)] + [y_shape[len(y_shape)-1]]
    assert len(new_x_shape) == len(new_y_shape) == num_batch_axes + 2
    # print("x_new_shape: {}".format(new_x_shape))
    # print("y_new_shape: {}".format(new_y_shape))

    # (b1, b2,..., bk, r, n)
    new_x = tf.reshape(x, new_x_shape)
    # (b1, b2,..., bk, s, n)
    new_y = tf.reshape(y, new_y_shape)

    y_perm = [i for i in range(len(new_y_shape))]
    y_perm[-1] = len(new_y_shape) -2
    y_perm[-2] = len(new_y_shape) -1

    # (b1, b2,..., bk, n, s)
    new_y = tf.transpose(new_y, y_perm)

    # Matmul (b1, b2,..., bk, r, n) and (b1, b2,..., bk, n, s)
    # (b1, b2,..., bk, r, s)
    out = tf.matmul(new_x, new_y)

    # (b1, b2,..., bk, r1, r2,..., rp, s1, s2,...,sq)
    original_shape = x_shape[0:num_batch_axes] + \
                     x_shape[num_batch_axes: len(x_shape)-1] + \
                     y_shape[num_batch_axes: len(y_shape)-1]

    # (b1, b2,..., bk, r1, r2,..., rp, s1, s2,...,sq)
    out = tf.reshape(out, original_shape)
    return out
