import tensorflow as tf

_EPSILON = 1e-8


def categorical_crossentropy(y_pred, y_true, from_logit=False):
    """
    :param y_pred: The predicted output
    :param y_true: The true output
    :param from_logit: If True, softmax has been computed for `y_pred`.
           If False, `y_pred` is still unnormalized.
    :return:
    """
    assert y_pred.get_shape().ndims == y_true.get_shape().ndims == 2

    if from_logit:
        return tf.nn.softmax_cross_entropy_with_logits(
            labels=y_true, logits=y_pred)
    else:
        y_pred = tf.nn.softmax(y_pred)
        _epsilon = tf.convert_to_tensor(_EPSILON, y_pred.dtype, name="EPSILON")
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1.0-_epsilon)
        return -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)


def sparse_categorical_crossentropy(y_pred, y_true, from_logit=False, num_classes=None):
    """
    :param y_pred: The predicted output
    :param y_true: The true output
    :param from_logit: If True, softmax has been computed for `y_pred`.
           If False, `y_pred` is still unnormalized.
    :param num_classes: The number of classes
    :return:
    """

    assert y_pred.get_shape().ndims == 2 and y_true.get_shape().ndims == 1

    if from_logit:
        return tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_true, logits=y_pred)
    else:
        y_pred = tf.nn.softmax(y_pred)
        _epsilon = tf.convert_to_tensor(_EPSILON, y_pred.dtype, name="EPSILON")
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1.0 - _epsilon)

        if num_classes is None:
            num_classes = y_pred.get_shape()[1]
            assert num_classes is not None

        y_true = tf.one_hot(y_true, num_classes, dtype=tf.float32)
        # Note that we have a minus sign here
        return -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)


def binary_crossentropy(y_pred, y_true, from_logit=False):
    """
    :param y_pred: The predicted output
    :param y_true: The true output
    :param from_logit: If True, softmax has been computed for `y_pred`.
           If False, `y_pred` is still unnormalized.
    :param num_classes: The number of classes
    :return:
    """
    assert y_pred.get_shape().ndims == 2 and y_true.get_shape().ndims == 2

    if from_logit:
        return tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred)
    else:
        y_pred = tf.nn.sigmoid(y_pred)
        _epsilon = tf.convert_to_tensor(_EPSILON, y_pred.dtype, name="EPSILON")
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1.0 - _epsilon)
        return -tf.reduce_sum(y_true * tf.log(y_pred) + (1.0-y_true) * tf.log(1.0-y_pred), axis=-1)


def l2_loss(params):
    assert isinstance(params, (tuple, list, set)), \
        "'params' must be a tuple/list/set. Found {}!".format(type(params))
    assert len(params) > 0, "'params' must be non-empty!"

    losses = []
    # Compute l2 loss for all trainable parameters in params
    for param in params:
        losses.append(tf.nn.l2_loss(param))
    return sum(losses, 0)


def l2_loss_for_scope(scope, excluded_params=None):
    # Compute l2 loss for all trainable parameters within a scope
    # excluded those in 'excluded_params'
    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    if excluded_params is not None:
        assert isinstance(excluded_params, (tuple, list, set)), \
            "If 'excluded_params' is not None, it must be a tuple/list/set. " \
            "Found {}!".format(type(excluded_params))

        excluded_params = set(excluded_params)

        selected_params = []
        for param in params:
            if not (param in excluded_params):
                selected_params.append(param)
    else:
        selected_params = params

    return l2_loss(selected_params)
