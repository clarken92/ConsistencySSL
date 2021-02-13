import numpy as np
from scipy import linalg


class Standardization:
    def __init__(self, x=None):
        self.mean = None
        self.std = None

        if x is not None:
            self.fit(x)

    def fit(self, x):
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0, ddof=0)

    def transform(self, x):
        assert self.mean is not None
        assert self.std is not None
        z = (x - self.mean) / np.clip(self.std, 1e-8, 1e8)
        return z

    def invert(self, x):
        assert self.mean is not None
        assert self.std is not None
        z = x * self.std + self.mean
        return z

    def save_npz(self, filename):
        np.savez_compressed(filename, mean=self.mean, std=self.std)

    @classmethod
    def load_npz(cls, filename):
        with np.load(filename, "r") as f:
            standard = cls()
            standard.mean = f['mean']
            standard.std = f['std']

        return standard


# From https://github.com/smlaine2/tempens
class ZCA:
    def __init__(self, x=None, eps=1e-5):
        self.mean = None
        self.zca_mat = None
        self.inv_zca_mat = None

        if x is not None:
            self.fit(x, eps)

    def fit(self, x, eps):
        shape = x.shape
        x = np.reshape(x, (shape[0], np.prod(shape[1:])))

        mean = np.mean(x, axis=0)
        x_centered = x - mean

        # We divided by the number of samples
        cov = np.dot(x_centered.T, x_centered) / x_centered.shape[0]

        # Singular Value Decomposition for the variance matrix
        U, S, V = linalg.svd(cov)
        tmp1 = np.dot(U, np.diag(1.0 / np.sqrt(S + eps)))
        tmp2 = np.dot(U, np.diag(np.sqrt(S + eps)))

        self.mean = mean
        self.zca_mat = np.dot(tmp1, U.T)
        self.inv_zca_mat = np.dot(tmp2, U.T)

    def transform(self, x):
        assert self.mean is not None
        assert self.zca_mat is not None

        shape = x.shape
        z = np.dot(x.reshape((shape[0], np.prod(shape[1:]))) - self.mean, self.zca_mat)
        return np.reshape(z, shape)

    def invert(self, x):
        assert self.mean is not None
        assert self.inv_zca_mat is not None
        shape = x.shape
        z = np.dot(x.reshape((shape[0], np.prod(shape[1:]))), self.inv_zca_mat) + self.mean
        return np.reshape(z, shape)

    def save_npz(self, filename):
        np.savez_compressed(filename, mean=self.mean,
                            zca_mat=self.zca_mat, inv_zca_mat=self.inv_zca_mat)

    @classmethod
    def load_npz(cls, filename):
        with np.load(filename, "r") as f:
            zca = cls()
            zca.mean = f['mean']
            zca.zca_mat = f['zca_mat']
            zca.inv_zca_mat = f['inv_zca_mat']

        return zca


# Check: http://cs231n.github.io/neural-networks-2/#datapre
# http://ufldl.stanford.edu/tutorial/unsupervised/PCAWhitening/
class PCA:
    def __init__(self, x_shape, num_comps=-1, whitening=True, x=None, eps=1e-5):
        self.x_shape = x_shape
        self.x_dim = np.prod(x_shape)

        if num_comps < 0:
            # If num_comps=-1, we use all principal components
            self.num_comps = self.x_dim
        else:
            assert num_comps < self.x_dim, "'x' has only {} features, " \
                "but 'num_comps'={}!".format(self.x_dim, num_comps)
            self.num_comps = num_comps

        self.whitening = whitening

        self.mean = None
        self.pca_mat = None
        self.inv_pca_mat = None

        if x is not None:
            self.fit(x, eps)

    def fit(self, x, eps=1e-5):
        shape = x.shape
        # (batch, dim)
        x = np.reshape(x, (shape[0], np.prod(shape[1:])))

        # (dim, )
        mean = np.mean(x, axis=0)
        x_centered = x - mean

        # We divided by the number of samples
        # (dim, dim)
        cov = np.dot(x_centered.T, x_centered) / x_centered.shape[0]

        # Singular Value Decomposition for the covariance matrix
        # (dim, dim)
        U, S, V = linalg.svd(cov)
        num_comps = self.num_comps

        pca_mat = U[:, :num_comps]
        if self.whitening:
            pca_mat = np.dot(pca_mat, np.diag(1.0 / np.sqrt(S[:num_comps] + eps)))

        if self.whitening:
            inv_pca_mat = np.dot(np.diag(np.sqrt(S[:num_comps] + eps)), U[:, :num_comps].T)
        else:
            inv_pca_mat = U[:, :num_comps].T

        self.mean = mean
        self.pca_mat = pca_mat
        self.inv_pca_mat = inv_pca_mat

    def transform(self, x):
        assert self.mean is not None
        assert self.pca_mat is not None
        shape = x.shape

        return np.dot(x.reshape((shape[0], np.prod(shape[1:]))) - self.mean, self.pca_mat)

    def invert(self, x):
        assert self.mean is not None
        assert self.inv_pca_mat is not None
        return (np.dot(x, self.inv_pca_mat) + self.mean).reshape(self.x_shape)

    def save_npz(self, filename):
        np.savez_compressed(filename, x_shape=self.x_shape,
                            num_comps=self.num_comps,
                            whitening=1 if self.whitening else 0,
                            mean=self.mean,
                            pca_mat=self.pca_mat, inv_pca_mat=self.inv_pca_mat)

    @classmethod
    def load_npz(cls, filename):
        with np.load(filename, "r") as f:
            pca = cls(f['x_shape'], num_comps=f['num_comps'],
                      whitening=True if f['whitening'] == 1 else False)
            pca.mean = f['mean']
            pca.pca_mat = f['zca_mat']
            pca.inv_pca_mat = f['inv_zca_mat']

        return pca
