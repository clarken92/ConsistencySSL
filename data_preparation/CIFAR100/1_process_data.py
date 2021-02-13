from __future__ import print_function

# Work around exception. Check here: https://stackoverflow.com/questions/35569042/ssl-certificate-verify-failed-with-python3
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from os.path import join

import tarfile
import numpy as np

from my_utils.python_utils.general import make_dir_if_not_exist
from my_utils.python_utils.image import uint8_to_float
from my_utils.python_utils.datasets import download_if_not_exist
from global_settings import RAW_DATA_DIR, PROCESSED_DATA_DIR


DATASET_URL = "https://www.cs.toronto.edu/~kriz/"
WIDTH = 32
HEIGHT = 32
CHANNELS = 3

DIR_RAW = join(RAW_DATA_DIR, "ComputerVision", "CIFAR100")
DIR_PROCESSED = join(PROCESSED_DATA_DIR, "ComputerVision", "CIFAR100")


# Check: https://www.cs.toronto.edu/~kriz/cifar.html
def unpickle(fo):
    import sys

    if sys.version_info.major >= 3:
        import pickle
        obj = pickle.load(fo, encoding='bytes')
        return obj
    else:
        import cPickle
        obj = cPickle.load(fo)
        return obj


# Extract the images and their labels
def extract_data_and_labels(file_path):
    """
    Extract the images into a 4D tensor [batch, height, width, channels].
    """
    print('Extracting {}!'.format(file_path))
    with tarfile.open(file_path) as tf:
        assert isinstance(tf, tarfile.TarFile)

        # TarInfo
        # print("members: {}".format(tf.getmembers()))
        print("names: {}".format(tf.getnames()))
        # cifar-100-python/meta
        # cifar-100-python/train
        # cifar-100-python/test

        # Meta data
        # ========================================== #
        f = tf.extractfile('cifar-100-python/meta')
        obj = unpickle(f)

        # 'fine_label_names', 'coarse_label_names'
        print("obj.keys: {}".format(list(obj.keys())))

        label_names = obj['fine_label_names'.encode(encoding='utf-8')]
        label_names = [b.decode(encoding='utf-8') for b in label_names]
        print("label_names: {}".format(label_names))

        label_names_coarse = obj['coarse_label_names'.encode(encoding='utf-8')]
        label_names_coarse = [b.decode(encoding='utf-8') for b in label_names_coarse]
        print("coarse_label_names: {}".format(label_names_coarse))
        # ========================================== #

        # Train data
        # ========================================== #
        f = tf.extractfile('cifar-100-python/train')
        obj = unpickle(f)

        # [filenames, batch_label, fine_labels, coarse_labels, data]
        print("\ntrain.keys: {}".format(list(obj.keys())))

        # Data
        # ------------------------------------------ #
        # We work on bytes, not string so we need to encode.
        # 'encode' means converting str to bytes. 'decode' means converting bytes to str.
        # 'encoding' is the encoding method (also use when decoding)
        x = obj['data'.encode(encoding='utf-8')]

        print("type(x): {}".format(type(x)))  # Numpy array
        print("x.shape: {}".format(x.shape))  # 50000 * 3072
        print("x.dtype: {}".format(x.dtype))  # uint8
        assert x.dtype == np.uint8, "x.dtype: {}".format(x.dtype)

        # 3072 = 3 * 32 * 32
        # The first 1024 entries contain the red channel values,
        # the next 1024 the green, and the final 1024 the blue.
        x = np.reshape(x, [50000, 3, 32, 32])
        x = np.transpose(x, [0, 2, 3, 1])
        train_data = x
        # ------------------------------------------ #

        # Labels
        # ------------------------------------------ #
        y = obj['fine_labels'.encode(encoding='utf-8')]
        print("type(y): {}".format(type(y)))         # list
        print("len(y): {}".format(len(y)))           # 50000
        print("type(y[0]): {}".format(type(y[0])))   # int
        train_labels = np.asarray(y, dtype=np.int32)

        y_coarse = obj['coarse_labels'.encode(encoding='utf-8')]
        train_labels_coarse = np.asarray(y_coarse, dtype=np.int32)
        # ------------------------------------------ #
        # ========================================== #

        # Test data
        # ========================================== #
        f = tf.extractfile('cifar-100-python/test')
        obj = unpickle(f)

        test_data = obj['data'.encode(encoding='utf-8')]
        assert test_data.dtype == np.uint8, "test_data.dtype: {}".format(test_data.dtype)

        test_data = np.reshape(test_data, [10000, 3, 32, 32])
        test_data = np.transpose(test_data, [0, 2, 3, 1])

        test_labels = obj['fine_labels'.encode(encoding='utf-8')]
        test_labels = np.asarray(test_labels, dtype=np.int32)

        test_labels_coarse = obj['coarse_labels'.encode(encoding='utf-8')]
        test_labels_coarse = np.asarray(test_labels_coarse, dtype=np.int32)
        # ========================================== #

        dataset = {
            'label_names': label_names,
            'label_names_coarse': label_names_coarse,
            'train_data': train_data,
            'train_labels': train_labels,
            'train_labels_coarse': train_labels_coarse,
            'test_data': test_data,
            'test_labels': test_labels,
            'test_labels_coarse': test_labels_coarse
        }

        return dataset


def plot_image(ax, img, title):
    ax.imshow(img, aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)


def main():
    raw_data_file = download_if_not_exist(DIR_RAW, "cifar-100-python.tar.gz", DATASET_URL)
    dataset = extract_data_and_labels(raw_data_file)

    train_data, train_labels, train_labels_lv2 = \
        dataset['train_data'], dataset['train_labels'], dataset['train_labels_coarse']
    test_data, test_labels, test_labels_lv2 = \
        dataset['test_data'], dataset['test_labels'], dataset['test_labels_coarse']
    label_names, label_names_lv2 = dataset['label_names'], dataset['label_names_coarse']

    modes = ['bytes', '0to1', 'm1p1']

    for mode in modes:
        print("\nCreate the '{}' version of CIFAR100!".format(mode))
        if mode == 'bytes':
            processed_train_data = train_data
            processed_test_data = test_data
        elif mode == '0to1':
            processed_train_data = uint8_to_float(train_data, pixel_inv_scale=255, pixel_shift=0)
            processed_test_data = uint8_to_float(test_data, pixel_inv_scale=255, pixel_shift=0)
        elif mode == 'm1p1':
            processed_train_data = uint8_to_float(train_data, pixel_inv_scale=127.5, pixel_shift=-1)
            processed_test_data = uint8_to_float(test_data, pixel_inv_scale=127.5, pixel_shift=-1)
        else:
            raise ValueError("Only support 'mode' in {}!".format(modes))

        save_dir = make_dir_if_not_exist(join(DIR_PROCESSED, mode))

        np.savez_compressed(join(save_dir, "train.npz"), x=processed_train_data,
                            y=train_labels, y_lv2=train_labels_lv2,
                            y_names=label_names, y_names_lv2=label_names_lv2)
        np.savez_compressed(join(save_dir, "test.npz"), x=processed_test_data,
                            y=test_labels, y_lv2=test_labels_lv2,
                            y_names=label_names, y_names_lv2=label_names_lv2)

        num_cols = 5
        train_idx = 100
        test_idx = 100

        # Uncomment if you want to show images
        # ---------------------------------- #
        '''
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, num_cols)
        for n in range(num_cols):
            if mode == "bytes" or mode == "0to1":
                plot_image(axes[0][n], processed_train_data[train_idx + n], title="train[{}]: {}".
                           format(train_idx + n, label_names[train_labels[train_idx + n]]))

                plot_image(axes[1][n], processed_test_data[test_idx + n], title="test[{}]: {}".
                           format(test_idx + n, label_names[test_labels[test_idx + n]]))

            elif mode == "m1p1":
                plot_image(axes[0][n], (processed_train_data[train_idx + n] + 1.0) / 2.0, title="train[{}]: {}".
                           format(train_idx + n, label_names[train_labels[train_idx + n]]))

                plot_image(axes[1][n], (processed_test_data[test_idx + n] + 1.0) / 2.0, title="test[{}]: {}".
                           format(test_idx + n, label_names[test_labels[test_idx + n]]))

        plt.show()
        '''
        # ---------------------------------- #

    from shutil import copyfile
    copyfile(join(DIR_PROCESSED, "bytes", "train.npz"),
             join(DIR_PROCESSED, "train.npz"))
    copyfile(join(DIR_PROCESSED, "bytes", "test.npz"),
             join(DIR_PROCESSED, "test.npz"))

if __name__ == "__main__":
    main()
