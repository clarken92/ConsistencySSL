from __future__ import print_function

# Work around exception. Check here: https://stackoverflow.com/questions/35569042/ssl-certificate-verify-failed-with-python3
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from os.path import join

import numpy as np
from scipy.io import loadmat

from my_utils.python_utils.general import make_dir_if_not_exist
from my_utils.python_utils.image import uint8_to_float
from my_utils.python_utils.datasets import download_if_not_exist
from global_settings import RAW_DATA_DIR, PROCESSED_DATA_DIR


DATASET_URL = "http://ufldl.stanford.edu/housenumbers/"
WIDTH = 32
HEIGHT = 32
CHANNELS = 3

DIR_RAW = join(RAW_DATA_DIR, "ComputerVision", "SVHN_CHECK")
DIR_PROCESSED = join(PROCESSED_DATA_DIR, "ComputerVision", "SVHN_CHECK")


# Extract the images and their labels
def extract_data_and_labels(file_path):
    """
    Extract the images into a 4D tensor [batch, height, width, channels].
    If 'scaled', images are rescaled from [0, 255] down to [0.0, 1.0]
    """
    print("\n" + "=" * 30)
    print('Extracting {}!'.format(file_path))
    assert file_path.endswith('.mat'), "Data file must be the original '.mat' file!"

    data_obj = loadmat(file_path)

    # 'X', 'y'
    print("keys: {}".format(list(data_obj.keys())))

    x = data_obj['X']
    print("")
    print("type(x): {}".format(type(x)))   # Numpy array
    print("x.shape: {}".format(x.shape))   # RGB image of shape [32, 32, 3, 73257] for train
    print("x.dtype: {}".format(x.dtype))   # uint8

    x = np.transpose(x, [3, 0, 1, 2])   # [73257, 32, 32, 3]

    y = data_obj['y']
    print("")
    print("type(y): {}".format(type(y)))   # Numpy array
    print("y.shape: {}".format(y.shape))   # RGB image of shape [73257, 1] for train
    print("y.dtype: {}".format(y.dtype))   # uint8

    y = np.squeeze(np.mod(y, 10), axis=1)  # [73257]
    print("y[:10]: {}".format(y[:10]))
    print("=" * 30)

    return x, y


def plot_image(ax, img, title):
    ax.imshow(img, aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)


def main():
    train_file = download_if_not_exist(DIR_RAW, "train_32x32.mat", DATASET_URL)
    test_file = download_if_not_exist(DIR_RAW, "test_32x32.mat", DATASET_URL)
    extra_file = download_if_not_exist(DIR_RAW, "extra_32x32.mat", DATASET_URL)

    train_data, train_labels = extract_data_and_labels(train_file)
    test_data, test_labels = extract_data_and_labels(test_file)
    extra_data, extra_labels = extract_data_and_labels(extra_file)
    label_names = None

    modes = ['bytes', '0to1', 'm1p1']

    for mode in modes:
        print("\nCreate the '{}' version of SVHN!".format(mode))
        if mode == 'bytes':
            processed_train_data = train_data
            processed_test_data = test_data
            processed_extra_data = extra_data
        elif mode == '0to1':
            processed_train_data = uint8_to_float(train_data, pixel_inv_scale=255, pixel_shift=0)
            processed_test_data = uint8_to_float(test_data, pixel_inv_scale=255, pixel_shift=0)
            processed_extra_data = uint8_to_float(extra_data, pixel_inv_scale=255, pixel_shift=0)
        elif mode == 'm1p1':
            processed_train_data = uint8_to_float(train_data, pixel_inv_scale=127.5, pixel_shift=-1)
            processed_test_data = uint8_to_float(test_data, pixel_inv_scale=127.5, pixel_shift=-1)
            processed_extra_data = uint8_to_float(extra_data, pixel_inv_scale=127.5, pixel_shift=-1)
        else:
            raise ValueError("Only support 'mode' in {}!".format(modes))

        save_dir = make_dir_if_not_exist(join(DIR_PROCESSED, mode))

        np.savez_compressed(join(save_dir, "train.npz"), x=processed_train_data,
                            y=train_labels, y_names=label_names)
        np.savez_compressed(join(save_dir, "test.npz"), x=processed_test_data,
                            y=test_labels, y_names=label_names)
        np.savez_compressed(join(save_dir, "extra.npz"), x=processed_extra_data,
                            y=extra_labels, y_names=label_names)

        num_cols = 5
        train_idx = 100
        test_idx = 100

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, num_cols)
        for n in range(num_cols):
            if mode == "bytes" or mode == "0to1":
                plot_image(axes[0][n], processed_train_data[train_idx + n], title="train[{}]: {}".
                           format(train_idx + n, train_labels[train_idx + n]))

                plot_image(axes[1][n], processed_test_data[test_idx + n], title="test[{}]: {}".
                           format(test_idx + n, test_labels[test_idx + n]))

            elif mode == "m1p1":
                plot_image(axes[0][n], (processed_train_data[train_idx + n] + 1.0) / 2.0, title="train[{}]: {}".
                           format(train_idx + n, train_labels[train_idx + n]))

                plot_image(axes[1][n], (processed_test_data[test_idx + n] + 1.0) / 2.0, title="test[{}]: {}".
                           format(test_idx + n, test_labels[test_idx + n]))

        plt.show()

    from shutil import copyfile
    copyfile(join(DIR_PROCESSED, "bytes", "train.npz"),
             join(DIR_PROCESSED, "train.npz"))
    copyfile(join(DIR_PROCESSED, "bytes", "test.npz"),
             join(DIR_PROCESSED, "test.npz"))
    copyfile(join(DIR_PROCESSED, "bytes", "extra.npz"),
             join(DIR_PROCESSED, "extra.npz"))

if __name__ == "__main__":
    main()
