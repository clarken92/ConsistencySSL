import tensorflow as tf

from ..shaping import mixed_shape


def conv_channel_maxout(x, pool_size, scope=None, reuse=None):
    with tf.variable_scope(scope or "conv_channel_maxout", reuse=reuse):
        batch, height, width, channels = mixed_shape(x)

        out = tf.reshape(x, [batch, height, width, channels // pool_size, pool_size])
        out = tf.reduce_max(out, axis=-1)

        return out


def max_pool_2D(x, pool_size, strides=None, padding="VALID", name=None):
    with tf.name_scope(name or "max_pool_2D"):
        x_shape = mixed_shape(x)
        assert len(x_shape) == 4, "'x' must be a 4D tensor!"

        if isinstance(pool_size, int):
            pool_size = (pool_size, pool_size)
        assert isinstance(pool_size, (list, tuple)) and len(pool_size) == 2, \
            "'pool_size' must be an int or a list/tuple of length 2!"

        if strides is None:
            strides = pool_size
        elif isinstance(strides, int):
            strides = (strides, strides)
        assert isinstance(strides, (list, tuple)) and len(strides) == 2, \
            "'strides' must be an int or a list/tuple of length 2!"

        return tf.nn.max_pool(x, ksize=[1, pool_size[0], pool_size[1], 1],
                              strides=[1, strides[0], strides[1], 1],
                              padding=padding, data_format="NHWC")


def avg_pool_2D(x, pool_size, strides=None, padding="VALID", name=None):
    with tf.name_scope(name or "avg_pool_2D"):
        x_shape = mixed_shape(x)
        assert len(x_shape) == 4, "'x' must be a 4D tensor!"

        if isinstance(pool_size, int):
            pool_size = (pool_size, pool_size)
        assert isinstance(pool_size, (list, tuple)) and len(pool_size) == 2, \
            "'pool_size' must be an int or a list/tuple of length 2!"

        if strides is None:
            strides = pool_size
        elif isinstance(strides, int):
            strides = (strides, strides)
        assert isinstance(strides, (list, tuple)) and len(strides) == 2, \
            "'strides' must be an int or a list/tuple of length 2!"

        return tf.nn.avg_pool(x, ksize=[1, pool_size[0], pool_size[1], 1],
                              strides=[1, strides[0], strides[1], 1],
                              padding=padding, data_format="NHWC")


def global_avg_pool_2D(x, keep_dims=True, name=None):
    with tf.name_scope(name or "global_avg_pool_2D"):
        x_shape = mixed_shape(x)
        assert len(x_shape) == 4, "'x' must be a 4D tensor!"

        return tf.reduce_mean(x, axis=[1, 2], keepdims=keep_dims)


def conv2d(x, filters, kernel_size, strides=1, padding="SAME", activation=None,
           use_bias=True, bias_initializer=tf.zeros_initializer(),
           weight_initializer=tf.glorot_uniform_initializer(),
           weight_regularizer=None, scope=None, reuse=None):

    with tf.variable_scope(scope or "conv2d", reuse=reuse):
        x_shape = mixed_shape(x)
        assert len(x_shape) == 4, "'x' must be a 4D tensor!"

        batch_size, inp_height, inp_width, inp_channels = x_shape

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        assert isinstance(kernel_size, (list, tuple)) and len(kernel_size) == 2, \
            "'kernel_size' must be an int or a list/tuple of length 2!"

        if isinstance(strides, int):
            strides = (strides, strides)
        assert isinstance(strides, (list, tuple)) and len(strides) == 2, \
            "'strides' must be an int or a list/tuple of length 2!"

        # IMPORTANT: Different from deconv2d, 'w' in conv2d must have out_channels AFTER inp_channels!
        w = tf.get_variable("w", shape=[kernel_size[0], kernel_size[1], inp_channels, filters],
                            dtype=tf.float32, initializer=weight_initializer, regularizer=weight_regularizer)

        output = tf.nn.conv2d(x, w, strides=[1, strides[0], strides[1], 1],
                              padding=padding.upper())

        if use_bias:
            b = tf.get_variable("b", shape=[filters], dtype=tf.float32,
                                initializer=bias_initializer)
            output += b

        if activation is not None:
            output = activation(output)

        return output


def deconv2d(x, output_shape, kernel_size, strides=1, padding="SAME", activation=None,
             use_bias=True, bias_initializer=tf.zeros_initializer(),
             weight_initializer=tf.glorot_uniform_initializer(),
             weight_regularizer=None, scope=None, reuse=None):

    with tf.variable_scope(scope or "deconv2d", reuse=reuse):
        x_shape = mixed_shape(x)
        assert len(x_shape) == 4, "'x' must be a 4D tensor!"

        batch_size, inp_height, inp_width, inp_channels = x_shape

        # assert isinstance(output_shape, (list, tuple)) and len(output_shape) == 3, \
        #     "'output_shape' must be a list/tuple of format (height, width, channels)!"
        out_height, out_width, out_channels = \
            output_shape[0], output_shape[1], output_shape[2]

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        assert isinstance(kernel_size, (list, tuple)) and len(kernel_size) == 2, \
            "'kernel_size' must be an int or a list/tuple of length 2!"

        if isinstance(strides, int):
            strides = (strides, strides)
        assert isinstance(strides, (list, tuple)) and len(strides) == 2, \
            "'strides' must be an int or a list/tuple of length 2!"

        # IMPORTANT: Different from conv2d, 'w' in deconv2d must have out_channels BEFORE inp_channels!
        w = tf.get_variable("w", shape=[kernel_size[0], kernel_size[1], out_channels, inp_channels],
                            dtype=tf.float32, initializer=weight_initializer, regularizer=weight_regularizer)

        output = tf.nn.conv2d_transpose(x, w, output_shape=[batch_size, out_height, out_width, out_channels],
                                        strides=[1, strides[0], strides[1], 1], padding=padding.upper())

        if use_bias:
            b = tf.get_variable("b", shape=[out_channels], dtype=tf.float32,
                                initializer=bias_initializer)
            output += b

        if activation is not None:
            output = activation(output)

        return output