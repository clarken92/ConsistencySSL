import tensorflow as tf
from my_utils.tensorflow_utils.activations import \
    exp_w_clip as exp_w_clip_, \
    log_w_clip as log_w_clip_, \
    sqrt_w_clip as sqrt_w_clip_

min4exp = -40.0
max4exp = 40.0
min4log = 1e-15
max4log = 1e30
min4sqrt = 1e-15
max4sqrt = 1e15

exp_w_clip = lambda x: exp_w_clip_(x, min=min4exp, max=max4exp)
log_w_clip = lambda x: log_w_clip_(x, min=min4log, max=max4log)
sqrt_w_clip = lambda x: sqrt_w_clip_(x, min=min4sqrt, max=max4sqrt)


EXISTING_LOG_ALPHAS = dict()

NOISY_WEIGHT_MODE = 1
STD_WEIGHT_MODE = 2
MASKED_WEIGHT_MODE = 3


def reset_existing_log_alphas():
    global EXISTING_LOG_ALPHAS
    EXISTING_LOG_ALPHAS = dict()


# Based on https://stackoverflow.com/questions/41695893/tensorflow-conditionally-add-variable-scope
def log_sigma2_variable(shape, ard_init=-10.):
    # log(sigma^2) is initialized with a very small value
    # sigma = e^-5 = 0.006738
    return tf.get_variable("log_sigma2", shape=shape,
                           initializer=tf.constant_initializer(ard_init))


def get_log_alpha(log_sigma2, w, container_scope):
    # alpha = sigma^2 / theta^2 (the paragraph between Eq.10 and Eq.11 in the paper)
    # Clip log_alpha within a reasonable range
    log_alpha = tf.clip_by_value(log_sigma2 - log_w_clip(tf.square(w)), -8.0, 8.0)
    log_alpha = tf.identity(log_alpha, name='log_alpha')

    if not (container_scope in EXISTING_LOG_ALPHAS):
        EXISTING_LOG_ALPHAS[container_scope] = log_alpha
        tf.add_to_collection('SVD_LOG_ALPHAS', log_alpha)

    return log_alpha


def l2_norm(x, axis, keepdims=False, eps=1e-12):
    norm = tf.sqrt(tf.maximum(tf.reduce_sum(x ** 2, axis=axis, keepdims=keepdims), eps))
    return norm


def l2_normalized(x, axis, eps=1e-12):
    norm = l2_norm(x, axis=axis, keepdims=True, eps=eps)
    x = x / norm
    return x


def fc_svd_wn(x, weight_mode, hid_dim, activation=None, use_bias=True, thresh=3,
              weight_initializer=tf.glorot_uniform_initializer(),
              bias_initializer=tf.zeros_initializer(),
              scope=None, reuse=None):

    with tf.variable_scope(scope or 'fc_svd_wn', reuse=reuse) as sc:
        x_shape = x.get_shape().as_list()
        assert len(x_shape) == 2, "x must be a 2D tensor of shape (batch, x_dim). " \
                                  "Found x.shape={}!".format(x_shape)
        x_dim = x_shape[1]

        w_shape = [x_dim, hid_dim]
        weight_init = weight_initializer(w_shape)
        weight_norm_init = l2_norm(weight_init, axis=[0], keepdims=False)

        v = tf.get_variable("v", dtype=tf.float32, initializer=weight_init)
        g = tf.get_variable("g", dtype=tf.float32, initializer=weight_norm_init)
        w = tf.reshape(g, [1, hid_dim]) * l2_normalized(v, axis=[0])

        # Init log_sigma2 with very small value
        log_sigma2 = log_sigma2_variable([x_dim, hid_dim])
        # log_alpha = get_log_alpha(log_sigma2, w)
        log_alpha = get_log_alpha(log_sigma2, w, sc.name)

        # At test time, we just mask the weight
        # If log_alpha < thresh, we mark 1 otherwise 0
        select_mask = tf.cast(tf.less(log_alpha, thresh), tf.float32)

        if isinstance(weight_mode, int):
            # print("'is_train' (stochastic) is a boolean!")
            if weight_mode == NOISY_WEIGHT_MODE:
                y = _fc_noisy(x, log_alpha, w)
            elif weight_mode == STD_WEIGHT_MODE:
                y = _fc_standard(x, w)
            elif weight_mode == MASKED_WEIGHT_MODE:
                y = _fc_masked(x, select_mask, w)
            else:
                raise ValueError("Do not support weight_mode={}!".format(weight_mode))
        else:
            y = tf.case(
                [(tf.equal(weight_mode, NOISY_WEIGHT_MODE),
                  lambda: _fc_noisy(x, log_alpha, w)),
                 (tf.equal(weight_mode, STD_WEIGHT_MODE),
                  lambda: _fc_standard(x, w)),
                 (tf.equal(weight_mode, MASKED_WEIGHT_MODE),
                  lambda: _fc_masked(x, select_mask, w))
                ])

        if use_bias:
            b = tf.get_variable("b", [hid_dim], initializer=bias_initializer)
            y = y + b

        if activation is not None:
            y = activation(y)

        return y


def fc_svd(x, weight_mode, hid_dim, activation=None, use_bias=True, thresh=3,
           weight_initializer=tf.glorot_uniform_initializer(),
           bias_initializer=tf.zeros_initializer(),
           scope=None, reuse=None):

    # NOTE: We should use glorot_uniform_initializer(), otherwise, the model won't work
    with tf.variable_scope(scope or 'fc_svd', reuse=reuse) as sc:
        x_shape = x.get_shape().as_list()
        assert len(x_shape) == 2, "x must be a 2D tensor of shape (batch, x_dim). " \
                                  "Found x.shape={}!".format(x_shape)
        x_dim = x_shape[1]

        # Actually, w is theta in the paper!
        w = tf.get_variable("w", [x_dim, hid_dim], initializer=weight_initializer)

        # Init log_sigma2 with very small value
        log_sigma2 = log_sigma2_variable([x_dim, hid_dim])
        # log_alpha = get_log_alpha(log_sigma2, w)
        log_alpha = get_log_alpha(log_sigma2, w, sc.name)

        # At test time, we just mask the weight
        # If log_alpha < thresh, we mark 1 otherwise 0
        select_mask = tf.cast(tf.less(log_alpha, thresh), tf.float32)

        if isinstance(weight_mode, int):
            # print("'is_train' (stochastic) is a boolean!")
            if weight_mode == NOISY_WEIGHT_MODE:
                y = _fc_noisy(x, log_alpha, w,)
            elif weight_mode == STD_WEIGHT_MODE:
                y = _fc_standard(x, w)
            elif weight_mode == MASKED_WEIGHT_MODE:
                y = _fc_masked(x, select_mask, w)
            else:
                raise ValueError("Do not support weight_mode={}!".format(weight_mode))
        else:
            y = tf.case(
                [(tf.equal(weight_mode, NOISY_WEIGHT_MODE),
                  lambda: _fc_noisy(x, log_alpha, w)),
                 (tf.equal(weight_mode, STD_WEIGHT_MODE),
                  lambda: _fc_standard(x, w)),
                 (tf.equal(weight_mode, MASKED_WEIGHT_MODE),
                  lambda: _fc_masked(x, select_mask, w))
                ])

        if use_bias:
            b = tf.get_variable("b", [hid_dim], initializer=bias_initializer)
            y = y + b

        if activation is not None:
            y = activation(y)

        return y


def _fc_noisy(x, log_alpha, w):
    # y = x w where w is deterministic weight
    mu = tf.matmul(x, w)  # Dot product between x and w (gamma in Eq. 17 in the paper)
    mu = tf.check_numerics(mu, "mu(fc_svd_noisy) is NaN!")
    # Dot product between (x^2) and (alpha * w^2) (delta in Eq. 17 in the paper)
    si = sqrt_w_clip(tf.matmul(tf.square(x), exp_w_clip(log_alpha) * tf.square(w)))
    si = tf.check_numerics(si, "si(fc_svd_noisy) is NaN!")
    # Noisy y
    y = mu + si * tf.random_normal(tf.shape(mu))
    y = tf.check_numerics(y, "y(fc_svd_noisy) is NaN!")

    return y


def _fc_standard(x, w):
    y = tf.matmul(x, w)
    # y = tf.check_numerics(y, "y(fc_svd_masked) is NaN!")
    return y


def _fc_masked(x, select_mask, w):
    # Dot product of x with masked weight
    y = tf.matmul(x, w * select_mask)
    # y = tf.check_numerics(y, "y(fc_svd_masked) is NaN!")
    return y


def conv2d_svd_wn(x, weight_mode, filters, kernel_size, strides=1, padding="SAME",
                  activation=None, use_bias=True, thresh=3,
                  weight_initializer=tf.glorot_uniform_initializer(),
                  bias_initializer=tf.zeros_initializer(),
                  scope=None, reuse=None):
    """
    svd_mode: An integer that specifies the SVD mode
    There are 3 modes:
        - Noisy (stochastic) weight: 1
        - Deterministic weight: 2
        - Deterministic weight with masking: 3

    The first mode is for training
    The last 2 modes are for testing
    """
    with tf.variable_scope(scope or "conv2d_svd_wn", reuse=reuse) as sc:
        x_shape = x.get_shape().as_list()
        assert len(x_shape) == 4, "x must be a 4D tensor of shape (batch, height, width, channels). " \
                                  "Found x.shape={}!".format(x_shape)
        inp_channels = x_shape[-1]

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        assert isinstance(kernel_size, (list, tuple)) and len(kernel_size) == 2, \
            "'kernel_size' must be an int or a list/tuple of length 2!"

        if isinstance(strides, int):
            strides = (strides, strides)
        assert isinstance(strides, (list, tuple)) and len(strides) == 2, \
            "'strides' must be an int or a list/tuple of length 2!"
        strides = [1, strides[0], strides[1], 1]

        w_shape = [kernel_size[0], kernel_size[1], inp_channels, filters]

        weight_init = weight_initializer(w_shape)
        weight_norm_init = l2_norm(weight_init, axis=[0, 1, 2], keepdims=False)

        v = tf.get_variable("v", dtype=tf.float32, initializer=weight_init)
        g = tf.get_variable("g", dtype=tf.float32, initializer=weight_norm_init)
        w = tf.reshape(g, [1, 1, 1, filters]) * l2_normalized(v, axis=[0])

        log_sigma2 = log_sigma2_variable(w_shape)
        # log_alpha = get_log_alpha(log_sigma2, w)
        log_alpha = get_log_alpha(log_sigma2, w, sc.name)

        select_mask = tf.cast(tf.less(log_alpha, thresh), tf.float32)

        if isinstance(weight_mode, int):
            # print("'is_train' (stochastic) is a boolean!")
            if weight_mode == NOISY_WEIGHT_MODE:
                y = _conv2d_noisy(x, log_alpha, w, padding=padding, strides=strides)
            elif weight_mode == STD_WEIGHT_MODE:
                y = _conv2d_standard(x, w, padding=padding, strides=strides)
            elif weight_mode == MASKED_WEIGHT_MODE:
                y = _conv2d_masked(x, select_mask, w, padding=padding, strides=strides)
            else:
                raise ValueError("Do not support weight_mode={}!".format(weight_mode))
        else:
            y = tf.case(
                [(tf.equal(weight_mode, NOISY_WEIGHT_MODE),
                  lambda: _conv2d_noisy(x, log_alpha, w, padding=padding, strides=strides)),
                 (tf.equal(weight_mode, STD_WEIGHT_MODE),
                  lambda: _conv2d_standard(x, w, padding=padding, strides=strides)),
                 (tf.equal(weight_mode, MASKED_WEIGHT_MODE),
                  lambda: _conv2d_masked(x, select_mask, w, padding=padding, strides=strides))
                ])

        if use_bias:
            b = tf.get_variable("b", [filters], initializer=bias_initializer)
            y = y + b

        if activation is not None:
            y = activation(y)

        return y


def conv2d_svd(x, weight_mode, filters, kernel_size, strides=1, padding="SAME",
               activation=None, use_bias=True, thresh=3,
               weight_initializer=tf.glorot_uniform_initializer(),
               bias_initializer=tf.zeros_initializer(),
               scope=None, reuse=None):
    """
    svd_mode: An integer that specifies the SVD mode
    There are 3 modes:
        - Noisy (stochastic) weight: 1
        - Deterministic weight: 2
        - Deterministic weight with masking: 3

    The first mode is for training
    The last 2 modes are for testing
    """
    with tf.variable_scope(scope or "conv2d_svd", reuse=reuse) as sc:
        x_shape = x.get_shape().as_list()
        assert len(x_shape) == 4, "x must be a 4D tensor of shape (batch, height, width, channels). " \
                                  "Found x.shape={}!".format(x_shape)
        inp_channels = x_shape[-1]

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        assert isinstance(kernel_size, (list, tuple)) and len(kernel_size) == 2, \
            "'kernel_size' must be an int or a list/tuple of length 2!"

        if isinstance(strides, int):
            strides = (strides, strides)
        assert isinstance(strides, (list, tuple)) and len(strides) == 2, \
            "'strides' must be an int or a list/tuple of length 2!"
        strides = [1, strides[0], strides[1], 1]

        w_shape = [kernel_size[0], kernel_size[1], inp_channels, filters]
        w = tf.get_variable("w", w_shape, initializer=weight_initializer)

        log_sigma2 = log_sigma2_variable(w_shape)
        # log_alpha = get_log_alpha(log_sigma2, w)
        log_alpha = get_log_alpha(log_sigma2, w, sc.name)

        select_mask = tf.cast(tf.less(log_alpha, thresh), tf.float32)

        if isinstance(weight_mode, int):
            # print("'is_train' (stochastic) is a boolean!")
            if weight_mode == NOISY_WEIGHT_MODE:
                y = _conv2d_noisy(x, log_alpha, w, padding=padding, strides=strides)
            elif weight_mode == STD_WEIGHT_MODE:
                y = _conv2d_standard(x, w, padding=padding, strides=strides)
            elif weight_mode == MASKED_WEIGHT_MODE:
                y = _conv2d_masked(x, select_mask, w, padding=padding, strides=strides)
            else:
                raise ValueError("Do not support weight_mode={}!".format(weight_mode))
        else:
            y = tf.case(
                [(tf.equal(weight_mode, NOISY_WEIGHT_MODE),
                  lambda: _conv2d_noisy(x, log_alpha, w, padding=padding, strides=strides)),
                 (tf.equal(weight_mode, STD_WEIGHT_MODE),
                  lambda: _conv2d_standard(x, w, padding=padding, strides=strides)),
                 (tf.equal(weight_mode, MASKED_WEIGHT_MODE),
                  lambda: _conv2d_masked(x, select_mask, w, padding=padding, strides=strides))
                ])

        if use_bias:
            b = tf.get_variable("b", [filters], initializer=bias_initializer)
            y = y + b

        if activation is not None:
            y = activation(y)

        return y


def _conv2d_noisy(x, log_alpha, w, padding, strides):
    log_alpha = tf.check_numerics(log_alpha, "log_alpha(conv2d_svd_noisy) is NaN!")
    w = tf.check_numerics(w, "w(conv2d_svd_noisy) is NaN!")

    # log_alpha = tf.Print(log_alpha, [tf.reduce_min(log_alpha), tf.reduce_max(log_alpha), tf.reduce_mean(log_alpha)],
    #                      message="min,max,mean(log_alpha (conv2d_svd_noisy)): ", summarize=1000)
    # w = tf.Print(w, [tf.reduce_min(w), tf.reduce_max(w), tf.reduce_mean(w)],
    #              message="min,max,mean(w (conv2d_svd_noisy)): ", summarize=1000)

    mu = tf.nn.conv2d(x, w, strides=strides, padding=padding)
    mu = tf.check_numerics(mu, "mu(conv2d_svd_noisy) is NaN!")

    si2 = tf.nn.conv2d(tf.square(x), exp_w_clip(log_alpha) * tf.square(w),
                       strides=strides, padding=padding)
    # si2 = tf.Print(si2, [tf.reduce_min(si2), tf.reduce_max(si2), tf.reduce_mean(si2)],
    #                message="min,max,mean(si2 (conv2d_svd_noisy)): ", summarize=1000)
    si2 = tf.check_numerics(si2, "si2(conv2d_svd_noisy) is NaN!")

    si = sqrt_w_clip(si2)
    # si = tf.Print(si, [tf.reduce_min(si), tf.reduce_max(si), tf.reduce_mean(si)],
    #               message="min,max,mean(si (conv2d_svd_noisy)): ", summarize=1000)
    si = tf.check_numerics(si, "si(conv2d_svd_noisy) is NaN!")

    y = mu + tf.random_normal(tf.shape(mu)) * si
    y = tf.check_numerics(y, "y(conv2d_svd_noisy) is NaN!")
    return y


def _conv2d_standard(x, w, padding, strides):
    y = tf.nn.conv2d(x, w, strides=strides, padding=padding)
    return y


def _conv2d_masked(x, select_mask, w, padding, strides):
    y = tf.nn.conv2d(x, w * select_mask, strides=strides, padding=padding)
    return y


# KL divergence approximation
def KL_qp_approx(log_alpha):
    k1, k2, k3 = 0.63576, 1.8732, 1.48695
    C = -k1
    neg_kl_approx = k1 * tf.nn.sigmoid(k2 + k3 * log_alpha) \
                    - 0.5 * log_w_clip(1 + exp_w_clip(-log_alpha)) + C
    return -tf.reduce_sum(neg_kl_approx)


# Handy function to keep track of sparsity
def sparsity(log_alphas, thresh=3):
    N_active, N_total = 0., 0.
    for la in log_alphas:
        m = tf.cast(tf.less(la, thresh), tf.float32)
        n_active = tf.reduce_sum(m)
        n_total = tf.cast(tf.reduce_prod(tf.shape(m)), tf.float32)
        N_active += n_active
        N_total += n_total
    return 1.0 - N_active / N_total


# Utility to collect variational dropout parameters
def collect_log_alphas(graph):
    node_defs = [n for n in graph.as_graph_def().node if 'log_alpha' in n.name]
    tensors = [graph.get_tensor_by_name(n.name + ":0") for n in node_defs]
    return tensors


def collect_log_alphas_v2(scopes):
    if isinstance(scopes, (list, tuple)):
        outputs = []
        for scope in scopes:
            outputs.extend(tf.get_collection('SVD_LOG_ALPHAS', scope=scope))
        return outputs
    else:
        return tf.get_collection('SVD_LOG_ALPHAS', scope=scopes)
