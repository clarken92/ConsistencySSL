from six import iteritems, moves
import tensorflow as tf


def custom_tf_scalar_summary(key, value, prefix=None):
    tag = key if prefix is None else "{}/{}".format(prefix, key)
    return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])


def custom_tf_scalar_summaries(keys_values, prefix=None):
    summaries = []
    for key, value in iteritems(keys_values):
        summaries.append(custom_tf_scalar_summary(key, value, prefix=prefix))
    return summaries


class ScalarSummarizer(object):
    def __init__(self, keys_and_aggr_types):
        """
        keys_and_aggr_types: A list of 2-tuples. Each tuple has the format (key, aggr_type)
        where 'aggr_type' can only be 'sum' or 'mean'
        """
        avail_aggr_types = ("mean", "sum")
        assert isinstance(keys_and_aggr_types, (tuple, list)), \
            "'keys_and_aggr_types' must be a list of 2-lists/tuples of the form (key, aggr_type)!"
        for n in moves.xrange(len(keys_and_aggr_types)):
            assert isinstance(keys_and_aggr_types[n], (tuple, list)) and len(keys_and_aggr_types[n]) == 2, \
                "The element {} of 'keys_and_aggr_types' must be a list/tuple of length 2!".format(n)
            assert isinstance(keys_and_aggr_types[n][0], (str, bytes)), \
                "The element {} of 'keys_and_aggr_types' must have 'key' to be str or bytes!".format(n)
            assert keys_and_aggr_types[n][1] in avail_aggr_types, \
                "The element {} of 'keys_and_aggr_types' must have 'aggr_type' in {}!".format(n, avail_aggr_types)

        self.keys_and_aggr_types = keys_and_aggr_types
        self._reset()

    def _reset(self):
        self.counts = [0 for _ in range(len(self.keys_and_aggr_types))]
        self.accumulations = [0.0 for _ in range(len(self.keys_and_aggr_types))]

    def accumulate(self, result_dict, batch_size):
        assert isinstance(result_dict, dict), "'result_dict' must be a dict!"
        for n, (key_, type_) in enumerate(self.keys_and_aggr_types):
            if key_ in result_dict:
                self.counts[n] += batch_size
                if type_ == "sum":
                    self.accumulations[n] += float(result_dict[key_])
                else:  # mean
                    self.accumulations[n] += float(result_dict[key_]) * batch_size

    def get_summaries_and_reset(self, summary_prefix):
        summaries = []
        results = {}

        for n, (key, _) in enumerate(self.keys_and_aggr_types):
            assert self.counts[n] > 0, "Counts for '{}' are not updated!".format(key)
            value = self.accumulations[n] / self.counts[n]
            results[key] = value

            summaries.append(custom_tf_scalar_summary(key, value, summary_prefix))

        self._reset()

        return summaries, results