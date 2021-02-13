from functools import partial
import tensorflow as tf


class empty_scope:
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        pass


def variable_scope_with_empty(scope, *args, **kwargs):
    return empty_scope() if scope is None else tf.variable_scope(scope, *args, **kwargs)


def name_scope_with_empty(name, *args, **kwargs):
    return empty_scope() if name is None else tf.name_scope(name, *args, **kwargs)


def assign_moving_average(x_old, x_new, momentum, name=None):
    with tf.name_scope(name or "assign_moving_average"):
        assert isinstance(x_old, tf.Variable), \
            "'x_old' must be a TF variable. Found {}!".format(type(x_old))
        y = momentum * x_old + (1 - momentum) * x_new
        return tf.assign(x_old, y)


# Use assign_add() instead of assign()
# assign_add is faster than assign().
# Check: https://github.com/tensorflow/tensorflow/issues/11514
def assign_moving_average_v2(x_old, x_new, momentum, name=None):
    with tf.name_scope(name or "assign_moving_average_v2"):
        assert isinstance(x_old, tf.Variable), \
            "'x_old' must be a TF variable. Found {}!".format(type(x_old))
        add_val = (1 - momentum) * (x_new - x_old)
        return tf.assign_add(x_old, add_val)


def get_params_from_scope(scope, trainable=True, excluded_subpattern=None, verbose=False):
    # Get parameters within a scope excluded those in 'excluded_params'
    if trainable:
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    else:
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

    if excluded_subpattern is not None:
        assert isinstance(excluded_subpattern, str), \
            "If 'excluded_subpattern' is not None, it must be a str. " \
            "Found {}!".format(type(excluded_subpattern))

        selected_params = []
        removed_params = []
        for param in params:
            if excluded_subpattern in param.name:
                removed_params.append(param)
            else:
                selected_params.append(param)
        if verbose:
            print("Removed params: {}".format(removed_params))
    else:
        selected_params = params

    return selected_params


def get_updates_from_scope(scope=None, excluded_subpattern=None, verbose=False):
    ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)

    if excluded_subpattern is not None:
        assert isinstance(excluded_subpattern, str), \
            "If 'excluded_subpattern' is not None, it must be a str. " \
            "Found {}!".format(type(excluded_subpattern))

        selected_ops = []
        removed_ops = []
        for op in ops:
            if excluded_subpattern in op.name:
                removed_ops.append(op)
            else:
                selected_ops.append(op)
        if verbose:
            print("Removed update ops: {}".format(removed_ops))
    else:
        selected_ops = ops

    return selected_ops


class FlexibleTemplate:
    def __init__(self, var_scope_name, fn, outer_name_scope_name=None, reuse=tf.AUTO_REUSE,
                 custom_getter=None, **fn_kwargs):
        self._vs_name = var_scope_name
        self._vs_reuse = reuse
        self._vs_custom_getter = custom_getter

        self._outer_ns_name = outer_name_scope_name

        self.fn = partial(fn, **fn_kwargs)

    def __call__(self, *fn_args, **fn_kwargs):
        with name_scope_with_empty(self._outer_ns_name):
            with tf.variable_scope(self._vs_name, reuse=self._vs_reuse, custom_getter=self._vs_custom_getter):
                return self.fn(*fn_args, **fn_kwargs)


class HyperParamUpdater:
    def __init__(self, variable_names, init_values, scope=None):
        assert isinstance(variable_names, (list, tuple)), \
            "'variable_names' must be a list/tuple. Found {}!".format(type(variable_names))
        self.variable_names = variable_names

        if init_values is None:
            init_values = [0.0] * len(variable_names)
        else:
            assert isinstance(init_values, (list, tuple)), \
                "'init_values' must be a list/tuple. Found {}!".format(type(variable_names))
            assert len(init_values) == len(variable_names)
        self.init_values = init_values

        self.variables = dict()
        self.placeholders = dict()
        self._update_ops = dict()

        with variable_scope_with_empty(scope):
            for var_name, init_val in zip(self.variable_names, self.init_values):
                var = tf.get_variable(var_name, shape=[], dtype=tf.float32, trainable=False,
                                      initializer=tf.constant_initializer(init_val))
                var_ph = tf.placeholder(dtype=tf.float32, shape=[], name=var_name + "_ph")
                update = tf.assign(var, var_ph)

                assert not (var_name in self.variables)
                self.variables[var_name] = var
                self.placeholders[var_name] = var_ph
                self._update_ops[var_name] = update

    def update(self, sess, feed_dict):
        updates = [self._update_ops[key] for key in feed_dict.keys()]
        feed_dict_ph = {self.placeholders[key]: feed_dict[key] for key in feed_dict.keys()}
        sess.run(updates, feed_dict=feed_dict_ph)

    def get_value(self, sess):
        return sess.run(self.variables)


# Version 2
# Here 'swa_params' can be created in advance
# This can be done in the model by calling a TF template with a new scope
class StochasticWeightAverage:
    def __init__(self, params, swa_params=None, param_scales=None, collection=None):
        self._num_updates = 0
        self._num_updates_ph = tf.placeholder(dtype=tf.float32, shape=[], name="num_updates")

        self._params = params

        if param_scales is None:
            self._param_scales = [None for _ in range(len(params))]
        else:
            assert len(param_scales) == len(params), "If 'param_scales' is not None, " \
                "its length ({}) must be equal to the length of 'params' ({})!"\
                .format(len(param_scales), len(params))
            self._param_scales = param_scales

        self.collection = "STOCHASTIC_WEIGHT_AVERAGE" if None else collection

        self.swa_params_dict = {}
        self.swa_updates = []
        self.swa_updates_first = []   # Update at first step, simply assign params to swa_params

        # Create 'swa_params' if it is not provided
        if swa_params is None:
            self.swa_params = []

            for param in params:
                # assert isinstance(param, tf.Variable), "type(param) = {}".format(type(param))
                with tf.variable_scope(param.op.name):
                    # Create swa_param
                    swa_param = tf.get_variable(
                        "SWA",  # shape=param.shape,
                        dtype=param._initial_value.dtype,
                        initializer=param._initial_value,
                        trainable=False,
                        collections=self.collection)
                    self.swa_params.append(swa_param)
                    self.swa_params_dict[param.op.name] = swa_param
        else:
            assert len(swa_params) == len(params), "len(swa_params) ({}) must " \
                "be equal to len(params) ({})!".format(len(swa_params), len(params))

            self.swa_params = swa_params
            for param, swa_param in zip(params, swa_params):
                self.swa_params_dict[param.op.name] = swa_param

        for i, (param, swa_param) in enumerate(zip(params, self.swa_params)):
            # Update
            if self._param_scales[i] is None:
                new_swa_param = (swa_param * (self._num_updates_ph - 1) + param) \
                                / self._num_updates_ph
            # log(w)
            elif self._param_scales[i] == "log":
                from my_utils.tensorflow_utils.activations import exp_w_clip, log_w_clip
                new_swa_param = (exp_w_clip(swa_param) * (self._num_updates_ph - 1) + exp_w_clip(param)) \
                                / self._num_updates_ph
                new_swa_param = log_w_clip(new_swa_param)
            elif self._param_scales[i] == "exp":
                from my_utils.tensorflow_utils.activations import exp_w_clip, log_w_clip
                new_swa_param = (log_w_clip(swa_param) * (self._num_updates_ph - 1) + log_w_clip(param)) \
                                / self._num_updates_ph
                new_swa_param = exp_w_clip(new_swa_param)
            else:
                raise ValueError("Do not support scale = '{}' for {}-th param!"
                                 .format(self._param_scales[i], i))

            swa_update = tf.assign(swa_param, new_swa_param)
            self.swa_updates.append(swa_update)

            # Update at first step
            swa_update_first = tf.assign(swa_param, param)
            self.swa_updates_first.append(swa_update_first)

    def update(self, sess):
        self._num_updates += 1

        if self._num_updates == 1:
            sess.run(self.swa_updates_first)
        else:
            sess.run(self.swa_updates, feed_dict={self._num_updates_ph: self._num_updates})

    @property
    def num_updates(self):
        return self._num_updates

    def swa_param_getter(self, getter, name, strict=False, *args, **kwargs):
        if strict:
            assert name in self.swa_params_dict, \
                "Parameter '{}' does not have SWA!".format(name)

        if name in self.swa_params_dict:
            return self.swa_params_dict[name]
        else:
            return getter(name, *args, **kwargs)

    '''
    def get_swa_param_values(self, sess, as_dict=False):
        if as_dict:
            swa_param_values = sess.run(self.swa_params)

            results = dict()
            for i in range(len(self.swa_params)):
                results[self.swa_params[i].name] = swa_param_values[i]

            return results
        else:
            return sess.run(self.swa_params)
    '''


def l2_normalized(a, axis, stop_norm_grad=True):
    norm = tf.sqrt(tf.reduce_sum(a ** 2, axis=axis, keepdims=True)) + 1e-15
    if stop_norm_grad:
        tf.stop_gradient(norm)

    a = a / norm
    return a
