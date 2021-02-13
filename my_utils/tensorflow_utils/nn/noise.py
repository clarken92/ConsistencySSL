import tensorflow as tf

from ..shaping import mixed_shape


def dropout(x, is_train, drop_rate, keep_axes, seed=None, name=None):
    '''
    For spatial 1D: (batch, steps, dim): noise_shape = [batch, 1, dim]
    For spatial 2D: (batch, height, width, channels): noise_shape = [batch, 1, 1, channels)

    The reason why we keep 'steps' or 'height', 'width' dimensions is that
    data along these dimensions are highly correlated.
    If we use dropout here, we will increase the variance, which leads to lower learning rate
    '''

    assert 0.0 < drop_rate < 1.0, "'drop_rate' must be between (0, 1). " \
        "Found {}!".format(drop_rate)

    noise_shape = mixed_shape(x)
    if isinstance(keep_axes, int):
        keep_axes = [keep_axes]

    for ax in keep_axes:
        noise_shape[ax] = 1

    with tf.name_scope(name or "dropout"):
        x_drop = tf.nn.dropout(x, keep_prob=(1.0 - drop_rate), noise_shape=noise_shape,
                               seed=seed, name='x_drop')
        # x_drop = tf.nn.dropout(x, rate=drop_rate, noise_shape=noise_shape,
        #                        seed=seed, name='x_drop')

        # output = tf.cond(is_train, lambda: x_drop, lambda: x)
        output = tf.where(is_train, x_drop, x)

        return output


def dropout_image(x, is_train, drop_rate=0.5, seed=None, name=None):
    ndims = x.get_shape().ndims
    assert ndims == 4, "'x' must be a 4D array of shape " \
        "(batch, height, width, channels). Found {}!".format(x.get_shape().as_list())

    keep_axes = [1, 2]
    return dropout(x, drop_rate=drop_rate, keep_axes=keep_axes,
                   is_train=is_train, seed=seed,
                   name="dropout_image" if name is None else name)


def dropout_2D(x, is_train, drop_rate, seed=None, name=None):
    return dropout_image(x, is_train=is_train, drop_rate=drop_rate, seed=seed, name=name)


def gauss_noise(x, is_train, std, name=None):
    with tf.name_scope(name or "gauss_noise"):
        eps = tf.random_normal(mixed_shape(x), 0.0, std)
        output = tf.cond(is_train, lambda: x + eps, lambda: x)
    return output
