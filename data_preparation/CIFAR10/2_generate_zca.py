from os.path import join

import numpy as np

from my_utils.python_utils.general import make_dir_if_not_exist
from my_utils.python_utils.data.normalization import ZCA
from my_utils.python_utils.image import uint8_to_binary_float
from global_settings import PROCESSED_DATA_DIR

DATA_DIR = join(PROCESSED_DATA_DIR, "ComputerVision", "CIFAR10", "bytes")
ZCA_DATA_DIR = make_dir_if_not_exist(join(PROCESSED_DATA_DIR, "ComputerVision", "CIFAR10_ZCA"))


def generate_zca_data():
    train_file = join(DATA_DIR, "train.npz")
    with np.load(train_file, "r") as f:
        train_x = uint8_to_binary_float(f['x'])
        train_y = f['y']
        y_names = f['y_names']

    test_file = join(DATA_DIR, "test.npz")
    with np.load(test_file, "r") as f:
        test_x = uint8_to_binary_float(f['x'])
        test_y = f['y']

    zca = ZCA(x=None, eps=1e-5)
    print("Fit train data!")
    zca.fit(train_x, eps=1e-5)
    print("Transform train data!")
    train_x = zca.transform(train_x)
    print("Transform test data!")
    test_x = zca.transform(test_x)
    assert train_x.shape == (50000, 32, 32, 3), "train_x.shape: {}".format(train_x.shape)
    assert test_x.shape == (10000, 32, 32, 3), "test_x.shape: {}".format(test_x.shape)

    zca_file = join(ZCA_DATA_DIR, "zca.npz")
    train_zca_file = join(ZCA_DATA_DIR, "train.npz")
    test_zca_file = join(ZCA_DATA_DIR, "test.npz")

    print("Save data!")
    zca.save_npz(zca_file)
    np.savez_compressed(train_zca_file, x=train_x, y=train_y, y_names=y_names)
    np.savez_compressed(test_zca_file, x=test_x, y=test_y, y_names=y_names)


# OK, similar to results provided by Keras
def check_zca():
    import matplotlib.pyplot as plt
    from my_utils.python_utils.image import scale_to_01

    # Train ZCA
    train_zca_file = join(ZCA_DATA_DIR, "train.npz")
    print("Check zca for train data!")
    with np.load(train_zca_file, "r") as f:
        train_x = f['x']

    m = 5
    n = 5
    fig, axes = plt.subplots(m, n)
    for i in range(0, m):
        for j in range(0, n):
            axes[i][j].imshow(scale_to_01(train_x[i * m + j], axis=None))
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])
    plt.show()

    # Test ZCA
    test_zca_file = join(ZCA_DATA_DIR, "test.npz")
    print("Check zca for test data!")
    with np.load(test_zca_file, "r") as f:
        test_x = f['x']

    m = 5
    n = 5
    fig, axes = plt.subplots(m, n)
    for i in range(0, m):
        for j in range(0, n):
            axes[i][j].imshow(scale_to_01(test_x[i * m + j], axis=None))
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])
    plt.show()


def main():
    generate_zca_data()
    check_zca()


if __name__ == "__main__":
    main()
