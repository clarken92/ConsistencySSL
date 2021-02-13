from six.moves.urllib import request
from os import makedirs
from os.path import join, exists
import numpy as np

from ..data.extraction import extract_data_by_total_counts, extract_data_by_class_proportions
from ..data.save_load import load_pkl


class SimpleDataset:
    def __init__(self):
        self.x = None
        self.y = None
        self.y_names = None

    @property
    def num_data(self):
        assert self.x is not None, "Data must be loaded in advance!"
        return len(self.x)

    @property
    def x_shape(self):
        assert self.x is not None, "Data must be loaded in advance!"
        return self.x.shape[1:]

    @property
    def y_shape(self):
        assert self.x is not None, "Data must be loaded in advance!"
        assert self.y is not None, "Data contains no labels!"
        return self.y.shape[1:]

    def preprocess_x(self):
        raise NotImplementedError

    def preprocess_y(self):
        raise NotImplementedError

    def load_mat_data(self, data_file):
        raise NotImplementedError

    def load_npz_data(self, data_file, verbose=True):
        if verbose:
            print("Loading npz data from file: [{}]".format(data_file))
        with np.load(data_file) as f:
            assert 'x' in f, "Cannot found a field named 'x' in file {}!".format(data_file)
            self.x = f['x']
            self.y = f.get('y', default=None)
            self.y_names = f.get('y_names', default=None)

    def load_pkl_data(self, data_file, verbose=True):
        if verbose:
            print("Loading pkl data from file: [{}]".format(data_file))
        obj = load_pkl(data_file)
        assert 'x' in obj, "Cannot found a field named 'x' in file {}!".format(data_file)
        self.x = obj['x']
        self.y = obj.get('y', default=None)
        self.y_names = obj.get('y_names', default=None)

    def fetch_batch(self, batch_ids, as_dict=False):
        xb = self.x[batch_ids]
        if self.y is not None:
            yb = self.y[batch_ids]
        else:
            yb = None

        return {'x': xb, 'y': yb} if as_dict else (xb, yb)

    def extract(self, selected_ids, remove=False):
        # Return a new 'SimpleDataset' objects containing 'selected_ids'
        # There is an option to decide whether we should remove these 'selected_ids' or not
        assert self.x is not None, "Data must be loaded in advance!"

        new_data_loader = SimpleDataset()
        new_data_loader.x = self.x[selected_ids]
        if self.y is not None:
            new_data_loader.y = self.y[selected_ids]
        new_data_loader.y_names = self.y_names

        if remove:
            self.x = np.delete(self.x, obj=selected_ids, axis=0)
            if self.y is not None:
                self.y = np.delete(self.y, obj=selected_ids, axis=0)

        return new_data_loader


class SimpleDataset4SSL(SimpleDataset):
    def __init__(self):
        SimpleDataset.__init__(self)

        # Boolean flags indicating which samples are labeled or not
        self.label_flag = None
        self.labeled_ids = None
        self.unlabeled_ids = None

    @property
    def num_labeled_data(self):
        assert self.labeled_ids is not None, "You must call 'create_ssl_data()' in advance!"
        return len(self.labeled_ids)

    @property
    def num_unlabeled_data(self):
        assert self.unlabeled_ids is not None, "You must call 'create_ssl_data()' in advance!"
        return len(self.unlabeled_ids)

    def create_ssl_data(self, labeled_spec, num_classes, shuffle=True, seed=None):
        # If 'labeled_spec' is an int, it is the total number of labeled data
        # If 'labeled_spec' is a float, it is the proportion of labeled data over the whole data
        # If 'labeled_spec' is a list of ints, it contains ids of labeled examples.

        assert self.x is not None
        assert self.y is not None

        # Total number of labeled examples
        if isinstance(labeled_spec, int):
            # print("There are {} labeled samples!".format(labeled_spec))
            num_labeled = labeled_spec
            assert num_labeled < self.num_data, "Number of labeled data is set to {} " \
                "but there are only {} data points!".format(num_labeled, self.num_data)

            labeled_ids = extract_data_by_total_counts(
                self.y, num_classes=num_classes,
                num_outputs=num_labeled, shuffle=shuffle, seed=seed)

        # Proportion of labeled examples (over all examples)
        elif isinstance(labeled_spec, float):
            print("{:.3f} data are labeled!".format(labeled_spec))
            assert 0.0 <= labeled_spec <= 1.0, \
                "'labeled_spec' must be in [0, 1] if float. Found {}!".format(labeled_spec)

            labeled_ids = extract_data_by_class_proportions(
                self.y, num_classes=num_classes,
                class_output_props=labeled_spec,
                shuffle=shuffle, seed=seed)

        # List of indices of labeled examples
        elif hasattr(labeled_spec, '__len__'):
            labeled_ids = labeled_spec

        else:
            raise ValueError("'labeled_spec' must be an int, a float, "
                             "or a list of sample ids. Found {}!".format(type(labeled_spec)))

        self.labeled_ids = labeled_ids

        label_flag = np.full([self.num_data], False, dtype=np.bool)
        label_flag[labeled_ids] = True
        self.label_flag = label_flag

        # Completely wrong
        # self.unlabeled_ids = self.y[np.logical_not(self.label_flag)]
        self.unlabeled_ids = np.arange(len(self.x))[np.logical_not(self.label_flag)]

    def fetch_batch(self, batch_ids, as_dict=False):
        x = self.x[batch_ids]
        y = self.y[batch_ids]
        label_flag = self.label_flag[batch_ids]

        return {'x': x, 'y': y, 'label_flag': label_flag} \
            if as_dict else (x, y, label_flag)

    def fetch_batch_v2(self, ids_l, ids_u, as_dict=False):
        # 'ids_l': Batch of relative ids for labeled data
        # 'ids_u': Batch of relative ids for unlabeled data
        true_ids_l = self.labeled_ids[ids_l]
        true_ids_u = self.unlabeled_ids[ids_u]

        xl = self.x[true_ids_l]
        yl = self.y[true_ids_l]
        xu = self.x[true_ids_u]
        yu = self.y[true_ids_u]

        return {'xl': xl, 'yl': yl, 'xu': xu, 'yu': yu} \
            if as_dict else (xl, yl, xu, yu)

# Download dataset
# ------------------------------------- #
def show_download_progress(count, block_size, total_size):
    print("\r{}% downloaded!".format(int(count * block_size * 100 / total_size)), end="")


def download_if_not_exist(dir_path, file_name, dataset_url):
    if not exists(dir_path):
        makedirs(dir_path)

    file_path = join(dir_path, file_name)
    if not exists(file_path):
        print("\n'{}' does not exist!".format(file_path))
        print("Start downloading '{}' from '{}'".format(file_name, dataset_url + file_name))
        file_path, _ = request.urlretrieve(
            dataset_url + file_name, file_path, reporthook=show_download_progress)
        print('\nSuccessfully downloaded {}!'.format(file_name))
    return file_path
# ------------------------------------- #
