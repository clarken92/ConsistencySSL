from __future__ import print_function
from io import open
import gzip
import pickle
from six import moves


def save_pkl(obj, file_path, verbose=False):
    assert file_path.endswith(".pkl") or file_path.endswith(".pkl.gz")
    if file_path.endswith(".pkl.gz"):
        f = gzip.open(file_path, 'wb')
    else:
        f = open(file_path, 'wb')
    pickle.dump(obj, f)
    f.close()

    if verbose:
        print("Saved to file [{}]!".format(file_path))


def save_pkl_multi_batches(obj, file_path, num_per_batch):
    assert isinstance(obj, list), "obj must be a list!"
    assert file_path.endswith(".pkl") or file_path.endswith(".pkl.gz")
    if file_path.endswith(".pkl.gz"):
        file_name, file_ext = file_path[:-len(".pkl.gz")], ".pkl.gz"
    else:
        file_name, file_ext = file_path[:-len(".pkl")], ".pkl"

    num_batches = len(obj) // num_per_batch
    for b in moves.xrange(num_batches):
        batch_file_path = file_name + "-part_%d" % b + file_ext
        save_pkl(obj[b * num_per_batch: (b+1) * num_per_batch], batch_file_path)

    num_remaining = len(obj) - num_batches * num_per_batch
    if num_remaining > 0:
        batch_file_path = file_name + "-part_%d" % num_batches + file_ext
        save_pkl(obj[num_batches * num_per_batch:], batch_file_path)


def load_pkl(file_path, verbose=False):
    assert file_path.endswith(".pkl") or file_path.endswith(".pkl.gz")
    if file_path.endswith(".pkl.gz"):
        f = gzip.open(file_path, 'rb')
    else:
        f = open(file_path, 'rb')

    obj = pickle.load(f)
    f.close()

    if verbose:
        print("Loaded from file [{}]!".format(file_path))

    return obj
