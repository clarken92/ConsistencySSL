import tensorflow as tf

from ..global_control import get_default_name, get_available_id


class BaseLayer(object):
    def __init__(self, scope=None, name=None):
        self.scope = scope or get_default_name(self)
        self.name = name or get_default_name(self)
        self.id = get_available_id()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("This function must be implemented!")

    def make_call_template(self, func_name, **kwargs):
        return tf.make_template(func_name, self.__call__, **kwargs)

