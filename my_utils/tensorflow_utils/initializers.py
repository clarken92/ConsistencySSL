import math
import numpy as np
import tensorflow as tf


def compute_fans(shape):
    """Computes the number of input and output units for a weight shape.

    Args:
        shape: Integer shape tuple or TF tensor shape.

    Returns:
        A tuple of scalars (fan_in, fan_out).
    """
    if len(shape) < 1:  # Just to avoid errors for constants.
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        # Assuming convolution kernels (2D, 3D, or more).
        # kernel shape: (..., input_depth, depth)
        receptive_field_size = 1.
        for dim in shape[:-2]:
            receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size

    return fan_in, fan_out


class NumpyVarianceScaling:
    """
    Initializer capable of adapting its scale to the shape of weights tensors.

    With `distribution="normal"`, samples are drawn from a truncated normal
    distribution centered on zero, with `stddev = sqrt(scale / n)`
    where n is:
        - number of input units in the weight tensor, if mode = "fan_in"
        - number of output units, if mode = "fan_out"
        - average of the numbers of input and output units, if mode = "fan_avg"

    With `distribution="uniform"`, samples are drawn from a uniform distribution
    within [-limit, limit], with `limit = sqrt(3 * scale / n)`.

    Args:
        scale: Scaling factor (positive float).
        mode: One of "fan_in", "fan_out", "fan_avg".
        distribution: Random distribution to use. One of "normal", "uniform".
        seed: A Python integer. Used to create random seeds. See
          @{tf.set_random_seed}
          for behavior.
        dtype: The data type. Only floating point types are supported.

    Raises:
        ValueError: In case of an invalid value for the "scale", mode" or
        "distribution" arguments.
    """

    def __init__(self,
                 scale=1.0,
                 mode="fan_in",
                 distribution="normal",
                 seed=None):
        if scale <= 0.:
            raise ValueError("`scale` must be positive float.")
        if mode not in {"fan_in", "fan_out", "fan_avg"}:
            raise ValueError("Invalid `mode` argument:", mode)
        distribution = distribution.lower()
        if distribution not in {"normal", "uniform"}:
            raise ValueError("Invalid `distribution` argument:", distribution)

        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.seed = seed
        self._rs = np.random.RandomState(seed)

    def __call__(self, shape):
        scale = self.scale

        fan_in, fan_out = compute_fans(shape)
        if self.mode == "fan_in":
            scale /= max(1., fan_in)
        elif self.mode == "fan_out":
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)

        if self.distribution == "normal":
            stddev = math.sqrt(scale)
            return self._rs.normal(0, scale=stddev, size=shape)
        else:
            limit = math.sqrt(3.0 * scale)
            return self._rs.uniform(-limit, limit, size=shape)


# https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/ops/init_ops.py
def get_weight_initializer(weight_init, **kwargs):
    if weight_init == 'uniform':
        return tf.random_uniform_initializer(**kwargs)
    elif weight_init == 'normal':
        return tf.random_normal_initializer(**kwargs)
    elif weight_init == 'truncated_normal':
        return tf.truncated_normal_initializer(**kwargs)
    elif weight_init == 'he_normal':
        return tf.variance_scaling_initializer(
            2.0, mode="fan_in", distribution="normal")
            # distribution="truncated_normal")
    elif weight_init == 'he_uniform':
        return tf.variance_scaling_initializer(
            2.0, mode="fan_in", distribution="uniform")
    elif weight_init == 'glorot_normal':
        return tf.variance_scaling_initializer(
            1.0, mode="fan_avg", distribution="normal")
            # distribution="truncated_normal")
    elif weight_init == 'glorot_uniform':
        return tf.variance_scaling_initializer(
            1.0, mode="fan_avg", distribution="uniform")
    else:
        raise ValueError("'weight_init' must be one of {}!".format(
            ('uniform', 'normal', 'truncated_normal',
             'he_normal', 'he_uniform', 'glorot_normal', 'glorot_uniform')))


def get_np_weight_init_values(mode, seed=None, **kwargs):
    if mode == 'uniform':
        pass
    elif mode == 'normal':
        pass
    elif mode == 'he_normal':
        return NumpyVarianceScaling(2.0, mode='fan_in', distribution='normal', seed=seed)
    elif mode == 'he_uniform':
        return NumpyVarianceScaling(2.0, mode='fan_in', distribution='uniform', seed=seed)
    elif mode == 'glorot_normal':
        return NumpyVarianceScaling(1.0, mode='fan_avg', distribution='normal', seed=seed)
    elif mode == 'glorot_uniform':
        return NumpyVarianceScaling(1.0, mode='fan_avg', distribution='uniform', seed=seed)
    else:
        raise ValueError("'weight_init' must be one of {}. Found {}!".format(
            ('uniform', 'normal', 'he_normal', 'he_uniform', 'glorot_normal', 'glorot_uniform'), mode))
