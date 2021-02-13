from six import moves, iteritems
from os.path import exists
import random
import shutil
import numpy as np


# ----------------------- Train Iteration ----------------------- #
# Use frequently
def iterate_data(data_size_or_ids, batch_size,
                 shuffle=False, seed=None, include_remaining=True):
    """
    V1.0: Stable running
    V1.1: Add seed and local RandomState
    :param data_size_or_ids:
    :param batch_size:
    :param shuffle:
    :param seed: None for complete randomisation
    :param include_remaining:
    :return:
    """
    if isinstance(data_size_or_ids, int):
        data_size = data_size_or_ids
        ids = list(range(data_size_or_ids))
    else:
        assert hasattr(data_size_or_ids, '__len__')
        ids = data_size_or_ids.tolist() if isinstance(data_size_or_ids, np.ndarray) \
            else list(data_size_or_ids)
        data_size = len(data_size_or_ids)

    rs = np.random.RandomState(seed)
    if shuffle:
        rs.shuffle(ids)
    nb_batch = len(ids) // batch_size

    for batch in moves.xrange(nb_batch):
        yield ids[batch * batch_size: (batch + 1) * batch_size]

    if include_remaining and nb_batch * batch_size < data_size:
        yield ids[nb_batch * batch_size:]


# Sampler
# --------------------------------------- #
# Use frequently
# This sampler repeatedly iterates over a dataset
class ContinuousIndexSampler(object):
    """
    V1.0: Stable running
    V1.1: Add seed and local RandomState
    """
    def __init__(self, data_size_or_ids, sample_size, shuffle=False, seed=None):
        self.sample_size = sample_size
        self.shuffle = shuffle
        self.seed = seed
        # It is OK to have a RandomState like this
        self._rs = np.random.RandomState(self.seed)

        if isinstance(data_size_or_ids, int):
            self.ids = list(range(data_size_or_ids))
        else:
            assert hasattr(data_size_or_ids, '__len__')
            self.ids = list(data_size_or_ids)

        self.sids = []
        while len(self.sids) < self.sample_size:
            self.sids += self._renew_ids()
        self.pointer = 0

    def _renew_ids(self):
        rids = [idx for idx in self.ids]
        if self.shuffle:
            # If seed is None, a completely new RandomState is created each time
            # rs = np.random.RandomState(self.seed)
            # rs.shuffle(rids)

            # We do not need to create a new RandomState each time
            self._rs.shuffle(rids)
        return rids

    def sample_ids(self):
        if self.pointer + self.sample_size > len(self.sids):
            self.sids = self.sids[self.pointer:] + self._renew_ids()
            while len(self.sids) < self.sample_size:
                self.sids += self._renew_ids()
            self.pointer = 0

        return_ids = self.sids[self.pointer: self.pointer + self.sample_size]
        self.pointer += self.sample_size
        return return_ids

    def sample_ids_continuous(self):
        while True:
            if self.pointer + self.sample_size > len(self.sids):
                self.sids = self.sids[self.pointer:] + self._renew_ids()
                while len(self.sids) < self.sample_size:
                    self.sids += self._renew_ids()
                self.pointer = 0

            return_ids = self.sids[self.pointer: self.pointer + self.sample_size]
            self.pointer += self.sample_size
            yield return_ids


class ContinuousIndexSamplerGroup(object):
    def __init__(self, *samplers):
        self.samplers = samplers

    def sample_group_of_ids(self):
        return (sampler.sample_ids() for sampler in self.samplers)
# --------------------------------------- #


class StoppingCondition(object):
    def __init__(self, best_improvement_steps=0, next_improvement_steps=0, mode='min'):
        """
        :param next_improvement_steps: Number of steps for next result improvement.
               If 0, stop immediately after seeing no improvement
               If n, after seeing no improvement, wait n more steps
        :param best_improvement_steps: Number of steps for best result improvement.
               If 0, stop immediately after seeing no improvement
               If n, after seeing no improvement, wait n more steps
        :param mode: min/max
        """

        self.next_imp_steps = next_improvement_steps
        self.best_imp_steps = best_improvement_steps

        assert mode in ['min', 'max'], "`mode` can only be either 'min' or 'max'!"
        if mode == 'min':
            self.comp_fn = lambda a, b: a <= b
            self.best_result = np.inf
            self.prev_result = np.inf
        else:
            self.comp_fn = lambda a, b: a >= b
            self.best_result = -np.inf
            self.prev_result = -np.inf

        self.curr_next_imp_step = 0
        self.curr_best_imp_step = 0

    def continue_running(self, result):
        # Improve over the best result
        if self.comp_fn(result, self.best_result):
            self.best_result = result
            self.curr_best_imp_step = 0
        else:
            self.curr_best_imp_step += 1

        # Improve over the previous result
        if self.comp_fn(result, self.prev_result):
            self.prev_result = result
            self.curr_next_imp_step = 0
        else:
            self.curr_next_imp_step += 1

        # No improvement over best or previous results after a predefined number of steps
        if self.curr_best_imp_step > self.best_imp_steps or \
           self.curr_next_imp_step > self.next_imp_steps:
            return False
        else:
            return True


class BestResultsTracker(object):
    def __init__(self, keys_and_cmpr_types, num_best):
        """
        keys: A list of strings
        num_best: Number of best results we want to keep track
        """

        avail_cmpr_types = ("less", "greater")
        assert isinstance(keys_and_cmpr_types, (tuple, list)), \
            "'keys_and_cmpr_types' must be a list of 2-lists/tuples of the form (key, cmpr_type)!"
        for n in moves.xrange(len(keys_and_cmpr_types)):
            assert isinstance(keys_and_cmpr_types[n], (tuple, list)) and len(keys_and_cmpr_types[n]) == 2, \
                "The element {} of 'keys_and_cmpr_types' must be a list/tuple of length 2!".format(n)
            assert isinstance(keys_and_cmpr_types[n][0], (str, bytes)), \
                "The element {} of 'keys_and_cmpr_types' must have 'key' to be str or bytes!".format(n)
            assert keys_and_cmpr_types[n][1] in avail_cmpr_types, \
                "The element {} of 'keys_and_cmpr_types' must have 'type' in {}!".format(n, avail_cmpr_types)

        self.keys_and_cmpr_type = keys_and_cmpr_types
        self.num_best = num_best

        # A dict that store best results
        # Key is the key we want to compare
        # Value is a 3-tuple:
        # The first element is the 'compr_type'
        # The second element is a list of length 'num_best' storing values
        # The last element is a list of length 'num_best' storing steps
        self._best_results = {key: (1.0 if cmpr_type == "greater" else -1.0,
                                    np.full(self.num_best, -np.inf, dtype=np.float32) \
                                        if cmpr_type == "greater" else \
                                    np.full(self.num_best, np.inf, dtype=np.float32),
                                    np.full(self.num_best, -1, dtype=np.int32))
                              for key, cmpr_type in keys_and_cmpr_types}

    @staticmethod
    def _better(x1, x2, cmpr_type):
        # Check whether x1 is better than x2 or not
        return ((x1 - x2) * cmpr_type) >= 0.0

    def check_and_update(self, results, step):
        assert isinstance(results, dict), "'results' must be a dict. Found {}!".format(type(results))

        # True: The key exist and its current value is better than stored values
        # False: The key exist and its current value is not as good as stored values
        # None: The key does not exist
        keys_checked = dict()

        for key, val in iteritems(results):
            _stored_results = self._best_results.get(key)

            if _stored_results is not None:
                keys_checked[key] = False
                cmpr_type, best_vals, best_steps = _stored_results
                for i in range(self.num_best):
                    if self._better(val, best_vals[i], cmpr_type):
                        best_vals[i + 1: self.num_best] = best_vals[i: self.num_best - 1]
                        best_steps[i + 1: self.num_best] = best_steps[i: self.num_best - 1]
                        best_vals[i] = val
                        best_steps[i] = step

                        keys_checked[key] = True
                        break

        return keys_checked

    def get_best_results(self):
        return self._best_results

    def set_best_results(self, results):
        self._validate_results(results)
        self._best_results = results

    def _validate_results(self, results):
        assert isinstance(results, dict), f"'results' must be a dict. Found {type(results)}!"
        for key, cmpr_type in self.keys_and_cmpr_type:
            val = results.get(key)

            assert val is not None, "'results' do not contain key {}!".format(key)
            assert isinstance(val, tuple) and len(val) == 3, \
                "'results[{}]' is not a tuple of length 3. Found {}!".format(key, type(val))

            if cmpr_type == "greater":
                assert val[0] == 1.0, "'cmpr_type' for key {0} is " \
                    "'greater' but val[0] in 'results[{0}]' is {1}!".format(key, val[0])
            else:
                assert val[0] == -1.0, "'cmpr_type' for key {0} is " \
                    "'less' but val[0] of 'results[{0}]' is {1}!".format(key, val[0])

            # assert isinstance(val[1], np.ndarray) and len(val[1]) == self.num_best \
            #     and val[1].dtype.name == 'float32', "val[1] of 'results[{}]' is {}!".format(key, val[1])
            #
            # assert isinstance(val[2], np.ndarray) and len(val[2]) == self.num_best \
            #     and val[2].dtype.name == 'int32', "val[2] of 'results[{}]' is {}!".format(key, val[2])

            # assert hasattr(val[1], '__len__'), f"'type(val[1])'={type(val[1])}"
            assert isinstance(val[1], np.ndarray), f"'type(val[1])'={type(val[1])}"
            assert val[1].dtype.name == 'float32', f"'val[1].dtype'={val[1].dtype.name}"
            assert len(val[1]) == self.num_best, f"'len(val[1])'={len(val[1])} while 'num_best'={self.num_best}!"
            # val[1] = np.asarray(val[1], dtype=np.float32)

            # assert hasattr(val[2], '__len__'), f"'type(val[2])'={type(val[2])}"
            assert isinstance(val[2], np.ndarray), f"'type(val[2])'={type(val[2])}"
            assert val[2].dtype.name == 'int32', f"'val[2].dtype'={val[2].dtype.name}"
            assert len(val[2]) == self.num_best, f"'len(val[2])'={len(val[2])} while 'num_best'={self.num_best}!"
            # val[2] = np.asarray(val[2], dtype=np.int32)
