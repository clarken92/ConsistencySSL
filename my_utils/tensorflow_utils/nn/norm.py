import tensorflow as tf
from ..general import assign_moving_average


# The 'momentum' and 'epsilon' values are default settings for DCGAN
# This is depricated in TF v2
# NOTE: For convolutional network, when you use batch norm, you should not use bias
# Check: https://stackoverflow.com/questions/46256747/can-not-use-both-bias-and-batch-normalization-in-convolution-layers
def batch_norm(x, is_train, axis=-1, momentum=0.9, epsilon=1e-5, scope=None, reuse=None):
    # axis=-1 if we want to do batch norm on channels of Conv2D
    return tf.layers.batch_normalization(
        x, training=is_train, axis=axis,
        momentum=momentum, epsilon=epsilon,
        center=True, scale=True,
        name=scope, reuse=reuse)


# This function is more accurate than batch_norm epsecially
# when there are MULTIPLE updates for the same 'moving_mean' and 'moving_variance'
# The reason is that this function uses tf.assign() while standard 'tf.layers.batch_normalization'
# uses tf.assign_add().
# Thus, when there are MULTIPLE updates, multiple values will be added to the current value
# Even batch_norm in Keras use 'tf.assign_sub()'
def bn(x, is_train, shift=True, scale=True, momentum=0.99, eps=1e-5,
       online_update=False, scope=None, reuse=None):

    # When x is an image (4D), we only consider channels
    # When x is a vector (2D), we only consider the features of x
    channels = x.get_shape().as_list()[-1]
    ndims = x.get_shape().ndims
    var_shape = [1] * (ndims - 1) + [channels]

    with tf.variable_scope(scope, 'bn', reuse=reuse):
        def training():
            # tf.nn.moments(x, axes, keep_dims)
            # The mean and variance are calculated by aggregating the contents of x across axes.
            # If x is 1D and axes = [0] this is just the mean and variance of a vector.

            # When using with convolutional filters with shape [batch, height, width, channel],
            # we should set axes=[0, 1, 2] to achieve GLOBAL NORMALIZATION
            m, v = tf.nn.moments(x, axes=list(range(ndims - 1)), keep_dims=True)

            # Update the mean and variance using moving average
            update_m = assign_moving_average(moving_mean, m, momentum, 'update_mean')
            update_v = assign_moving_average(moving_var, v, momentum, 'update_var')

            if online_update:
                with tf.control_dependencies([update_m, update_v]):
                    # tf.rsqrt: reciprocal of the square root
                    # y = 1/sqrt(x)
                    output = (x - m) * tf.rsqrt(v + eps)
            else:
                tf.add_to_collection('update_ops', update_m)
                tf.add_to_collection('update_ops', update_v)
                output = (x - m) * tf.rsqrt(v + eps)

            return output

        def testing():
            m, v = moving_mean, moving_var
            output = (x - m) * tf.rsqrt(v + eps)
            return output

        # Initialize moving mean and variance
        # trainable=False, only for updating
        moving_mean = tf.get_variable('moving_mean', var_shape, initializer=tf.zeros_initializer(), trainable=False)
        moving_var = tf.get_variable('moving_var', var_shape, initializer=tf.ones_initializer(), trainable=False)

        if isinstance(is_train, bool):
            output = training() if is_train else testing()
        else:
            output = tf.cond(is_train, training, testing)

        if scale:
            # trainable=True, adjusting the scale after normalization
            output *= tf.get_variable('gamma', var_shape, initializer=tf.ones_initializer)

        if shift:
            # trainable=False, adjusting the ship after normalization
            output += tf.get_variable('beta', var_shape, initializer=tf.zeros_initializer)

    return output