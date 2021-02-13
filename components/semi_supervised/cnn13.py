from functools import partial
import tensorflow as tf

from my_utils.tensorflow_utils.layers import BaseLayer
from my_utils.tensorflow_utils.initializers import get_weight_initializer
from my_utils.tensorflow_utils.shaping import flatten_right_from
from my_utils.tensorflow_utils.nn import linear, fc as fc_, \
    conv2d as conv2d_, deconv2d as deconv2d_, \
    batch_norm as batch_norm_, dropout_2D, \
    max_pool_2D, avg_pool_2D, gauss_noise


# 9310gaurav (CNN-13)
# ======================================= #
# V1.0: Similar to the CNN-13 model used by MeanTeacher
# V1.1: Add 'use_gaussian_noise'
# V1.2: Allow us to disable some stochastic layers such as Gaussian Noise or Dropout
class MainClassifier_9310gaurav(BaseLayer):
    def __init__(self, num_classes=10, bn_momentum=0.999, use_gauss_noise=True, scope=None):
        BaseLayer.__init__(self, scope)
        self.num_classes = num_classes
        self.bn_momentum = bn_momentum
        self.use_gauss_noise = use_gauss_noise

    def __call__(self, x, is_train, stochastic=None, scope=None, reuse=None):
        weight_init = get_weight_initializer('he_normal')
        conv2d = partial(conv2d_, weight_initializer=weight_init, activation=None)
        fc = partial(fc_, weight_initializer=weight_init, activation=None)
        bn = partial(batch_norm_, momentum=self.bn_momentum, epsilon=1e-8)
        act = partial(tf.nn.leaky_relu, alpha=0.1)

        if stochastic is None:
            stochastic = is_train

        with tf.variable_scope(self.scope, reuse=reuse):
            x_shape = x.shape.as_list()
            assert len(x_shape) == 4 and x_shape[1] == x_shape[2] == 32 and x_shape[3] == 3, \
                "x.shape={}".format(x_shape)

            h = x

            if self.use_gauss_noise:
                h = gauss_noise(h, stochastic, std=0.15)

            # conv 1
            # --------------------------------------- #
            with tf.variable_scope("conv_1a"):
                # (32, 32, 3) => (32, 32, 128)
                h = conv2d(h, filters=128, kernel_size=3, strides=1,
                           padding="SAME", use_bias=False)
                h = bn(h, is_train)
                h = act(h)

            with tf.variable_scope("conv_1b"):
                # (32, 32, 128) => (32, 32, 128)
                h = conv2d(h, filters=128, kernel_size=3, strides=1,
                           padding="SAME", use_bias=False)
                h = bn(h, is_train)
                h = act(h)

            with tf.variable_scope("conv_1c"):
                # (32, 32, 128) => (32, 32, 128)
                h = conv2d(h, filters=128, kernel_size=3, strides=1,
                           padding="SAME", use_bias=False)
                h = bn(h, is_train)
                h = act(h)
            # --------------------------------------- #

            # (32, 32, 128) => (16, 16, 128)
            h = max_pool_2D(h, pool_size=2)
            h = dropout_2D(h, stochastic, drop_rate=0.5)

            # conv 2
            # --------------------------------------- #
            with tf.variable_scope("conv_2a"):
                # (16, 16, 128) => (16, 16, 256)
                h = conv2d(h, filters=256, kernel_size=3, strides=1,
                           padding="SAME", use_bias=False)
                h = bn(h, is_train)
                h = act(h)

            with tf.variable_scope("conv_2b"):
                # (16, 16, 256) => (16, 16, 256)
                h = conv2d(h, filters=256, kernel_size=3, strides=1,
                           padding="SAME", use_bias=False)
                h = bn(h, is_train)
                h = act(h)

            with tf.variable_scope("conv_2c"):
                # (16, 16, 256) => (16, 16, 256)
                h = conv2d(h, filters=256, kernel_size=3, strides=1,
                           padding="SAME", use_bias=False)
                h = bn(h, is_train)
                h = act(h)
            # --------------------------------------- #

            # (16, 16, 256) => (8, 8, 256)
            h = max_pool_2D(h, pool_size=2)
            h = dropout_2D(h, stochastic, drop_rate=0.5)

            # conv 3
            # --------------------------------------- #
            with tf.variable_scope("conv_3a"):
                # (8, 8, 256) => (6, 6, 512)
                h = conv2d(h, filters=512, kernel_size=3, strides=1,
                           padding="VALID", use_bias=False)
                h = bn(h, is_train)
                h = act(h)

            with tf.variable_scope("conv_3b"):
                # (6, 6, 512) => (6, 6, 256)
                h = conv2d(h, filters=256, kernel_size=1, strides=1,
                           padding="VALID", use_bias=False)
                h = bn(h, is_train)
                h = act(h)

            with tf.variable_scope("conv_3c"):
                # (6, 6, 256) => (6, 6, 128)
                h = conv2d(h, filters=128, kernel_size=1, strides=1,
                           padding="VALID", use_bias=False)
                h = bn(h, is_train)
                h = act(h)
            # --------------------------------------- #

            # (1, 1, 128)
            h = avg_pool_2D(h, pool_size=6)
            # (128,)
            h = flatten_right_from(h, 1)

            y_logit = fc(h, hid_dim=self.num_classes, use_bias=True)
            y_prob = tf.nn.softmax(y_logit)

            return {'logit': y_logit, 'prob': y_prob, 'hid': h}


# Very similar to 'MainClassifier4PiModel_9310gaurav' except that
# we use Variational Dropout in every layer (conv + fc)
class MainClassifier_VD_9310gaurav(BaseLayer):
    def __init__(self, num_classes=10, bn_momentum=0.999, use_gauss_noise=True, thresh=3.0, scope=None):
        BaseLayer.__init__(self, scope)
        self.num_classes = num_classes
        self.bn_momentum = bn_momentum
        self.use_gauss_noise = use_gauss_noise
        self.thresh = thresh

    def __call__(self, x, is_train, weight_mode, stochastic=None, scope=None, reuse=None):
        from my_utils.tensorflow_utils.bayesian.svd import \
            conv2d_svd as conv2d_svd_, fc_svd as fc_svd_
        print("weight_mode: {}".format(weight_mode))

        weight_init = get_weight_initializer('he_normal')
        conv2d = partial(conv2d_svd_, weight_mode=weight_mode,
                         weight_initializer=weight_init,
                         thresh=self.thresh, activation=None)
        fc = partial(fc_svd_, weight_mode=weight_mode,
                     weight_initializer=weight_init,
                     thresh=self.thresh, activation=None)
        bn = partial(batch_norm_, momentum=self.bn_momentum, epsilon=1e-8)
        act = partial(tf.nn.leaky_relu, alpha=0.1)

        if stochastic is None:
            stochastic = is_train

        with tf.variable_scope(self.scope, reuse=reuse):
            x_shape = x.shape.as_list()
            assert len(x_shape) == 4 and x_shape[1] == x_shape[2] == 32 and x_shape[3] == 3, \
                "x.shape={}".format(x_shape)

            h = x

            if self.use_gauss_noise:
                h = gauss_noise(h, stochastic, std=0.15)

            # conv 1
            # --------------------------------------- #
            with tf.variable_scope("conv_1a"):
                # (32, 32, 3) => (32, 32, 128)
                h = conv2d(h, filters=128, kernel_size=3, strides=1,
                           padding="SAME", use_bias=False)
                h = bn(h, is_train)
                h = act(h)

            with tf.variable_scope("conv_1b"):
                # (32, 32, 128) => (32, 32, 128)
                h = conv2d(h, filters=128, kernel_size=3, strides=1,
                           padding="SAME", use_bias=False)
                h = bn(h, is_train)
                h = act(h)

            with tf.variable_scope("conv_1c"):
                # (32, 32, 128) => (32, 32, 128)
                h = conv2d(h, filters=128, kernel_size=3, strides=1,
                           padding="SAME", use_bias=False)
                h = bn(h, is_train)
                h = act(h)
            # --------------------------------------- #

            # (32, 32, 128) => (16, 16, 128)
            h = max_pool_2D(h, pool_size=2)
            # Remove dropout when using SVD!
            # h = dropout_2D(h, is_train, drop_rate=0.5)

            # conv 2
            # --------------------------------------- #
            with tf.variable_scope("conv_2a"):
                # (16, 16, 128) => (16, 16, 256)
                h = conv2d(h, filters=256, kernel_size=3, strides=1,
                           padding="SAME", use_bias=False)
                h = bn(h, is_train)
                h = act(h)

            with tf.variable_scope("conv_2b"):
                # (16, 16, 256) => (16, 16, 256)
                h = conv2d(h, filters=256, kernel_size=3, strides=1,
                           padding="SAME", use_bias=False)
                h = bn(h, is_train)
                h = act(h)

            with tf.variable_scope("conv_2c"):
                # (16, 16, 256) => (16, 16, 256)
                h = conv2d(h, filters=256, kernel_size=3, strides=1,
                           padding="SAME", use_bias=False)
                h = bn(h, is_train)
                h = act(h)
            # --------------------------------------- #

            # (16, 16, 256) => (8, 8, 256)
            h = max_pool_2D(h, pool_size=2)
            # Remove dropout when using SVD!
            # h = dropout_2D(h, is_train, drop_rate=0.5)

            # conv 3
            # --------------------------------------- #
            with tf.variable_scope("conv_3a"):
                # (8, 8, 256) => (6, 6, 512)
                h = conv2d(h, filters=512, kernel_size=3, strides=1,
                           padding="VALID", use_bias=False)
                h = bn(h, is_train)
                h = act(h)

            with tf.variable_scope("conv_3b"):
                # (6, 6, 512) => (6, 6, 256)
                h = conv2d(h, filters=256, kernel_size=1, strides=1,
                           padding="VALID", use_bias=False)
                h = bn(h, is_train)
                h = act(h)

            with tf.variable_scope("conv_3c"):
                # (6, 6, 256) => (6, 6, 128)
                h = conv2d(h, filters=128, kernel_size=1, strides=1,
                           padding="VALID", use_bias=False)
                h = bn(h, is_train)
                h = act(h)
            # --------------------------------------- #

            # (1, 1, 128)
            h = avg_pool_2D(h, pool_size=6)
            # (128, )
            h = flatten_right_from(h, 1)

            y_logit = fc(h, hid_dim=self.num_classes, use_bias=True)
            y_prob = tf.nn.softmax(y_logit)

            return {'logit': y_logit, 'prob': y_prob, 'hid': h}
# ======================================= #
