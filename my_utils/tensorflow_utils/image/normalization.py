import numpy as np
from scipy import linalg
import tensorflow as tf

from ..shaping import mixed_shape, flatten_right_from


# Check: http://cs231n.github.io/neural-networks-2/#datapre
# http://ufldl.stanford.edu/tutorial/unsupervised/PCAWhitening/
# https://www.kdnuggets.com/2018/10/preprocessing-deep-learning-covariance-matrix-image-whitening.html/3
class Standardization:
    def __init__(self, image_shape, scope=None, x=None):
        self.scope = scope or "Standardization"

        assert len(image_shape) == 3, "image_shape: {}".format(image_shape)
        self.image_shape = tuple(image_shape)

        if x is not None:
            assert tuple(x.shape[1:]) == self.image_shape, \
                "image_shape={} while x.shape={}".format(self.image_shape, x.shape)
            results = self.fit_np(x)

            with tf.variable_scope(self.scope):
                self.mean = tf.get_variable("mean", shape=image_shape, dtype=tf.float32,
                                            initializer=tf.constant_initializer(results['mean']))
                self.std = tf.get_variable("std", shape=image_shape, dtype=tf.float32,
                                           initializer=tf.constant_initializer(results['std']))
            self._built = True

        else:
            with tf.variable_scope(self.scope):
                self.mean = tf.get_variable("mean", shape=image_shape, dtype=tf.float32,
                                            initializer=tf.zeros_initializer())
                self.std = tf.get_variable("std", shape=image_shape, dtype=tf.float32,
                                           initializer=tf.ones_initializer())
            self._built = False

    def fit_np(self, x):
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)

        return {
            'mean': mean,
            'std': std,
        }

    def fit_and_init_np(self, sess, x):
        assert not self._built, "The model's parameters must have not been built!"
        results = self.fit_np(x)

        sess.run([tf.assign(self.mean, results['mean']),
                  tf.assign(self.std, results['std'])])
        self._built = True

    def transform(self, x, name=None):
        assert self._built, "The model's parameters must have been built!"
        with tf.name_scope(name or "standardization_transform"):
            z = (x - self.mean) / tf.clip_by_value(self.std, 1e-8, 1e8)
        return z

    def invert(self, x, name=None):
        assert self._built, "The model's parameters must have been built!"
        with tf.name_scope(name or "standardization_invert"):
            z = x * self.std + self.mean
        return z


# ZCA whitening only differ from PCA whitening in that
# it add a small epsilon to S and transform the data back to the original space
# Check: https://stats.stackexchange.com/questions/117427/what-is-the-difference-between-zca-whitening-and-pca-whitening
class ZCA:
    def __init__(self, image_shape, scope=None, x=None, eps=1e-5):
        self.scope = scope or "ZCA"

        assert len(image_shape) == 3, "image_shape: {}".format(image_shape)
        self.image_shape = tuple(image_shape)
        self.dim = np.prod(self.image_shape)

        if x is not None:
            assert tuple(x.shape[1:]) == self.image_shape, \
                "image_shape={} while x.shape={}".format(self.image_shape, x.shape)
            results = self.fit_np(x, eps=eps)

            with tf.variable_scope(self.scope):
                self.eigen_col_vectors = tf.get_variable("eigen_col_vectors", shape=[self.dim, self.dim], dtype=tf.float32,
                                                         initializer=tf.constant_initializer(results['eigen_col_vectors']))
                self.eigen_values = tf.get_variable("eigen_values", shape=[self.dim], dtype=tf.float32,
                                                    initializer=tf.constant_initializer(results['eigen_values']))
                self.zca_mat = tf.get_variable("zca_mat", shape=[self.dim, self.dim], dtype=tf.float32,
                                               initializer=tf.constant_initializer(results['zca_mat']))
                self.inv_zca_mat = tf.get_variable("inv_zca_mat", shape=[self.dim, self.dim], dtype=tf.float32,
                                                   initializer=tf.constant_initializer(results['inv_zca_mat']))
                self.mean = tf.get_variable("mean", shape=[self.dim], dtype=tf.float32,
                                            initializer=tf.constant_initializer(results['mean']))
                self._built = True
        else:
            with tf.variable_scope(self.scope):
                self.eigen_col_vectors = tf.get_variable("eigen_col_vectors", shape=[self.dim, self.dim], dtype=tf.float32,
                                                         initializer=tf.zeros_initializer())
                self.eigen_values = tf.get_variable("eigen_values", shape=[self.dim], dtype=tf.float32,
                                                    initializer=tf.zeros_initializer())
                self.zca_mat = tf.get_variable("zca_mat", shape=[self.dim, self.dim], dtype=tf.float32,
                                               initializer=tf.ones_initializer())
                self.inv_zca_mat = tf.get_variable("inv_zca_mat", shape=[self.dim, self.dim], dtype=tf.float32,
                                                   initializer=tf.ones_initializer())
                self.mean = tf.get_variable("mean", shape=[self.dim], dtype=tf.float32,
                                            initializer=tf.zeros_initializer())

            self._built = False

    def fit_np(self, x, eps=1e-5):
        shape = x.shape
        x = np.reshape(x, [shape[0], self.dim])

        # (dim, )
        mean = np.mean(x, axis=0)
        x_centered = x - mean

        # (dim, dim)
        var = np.dot(x_centered.T, x_centered) / shape[0]

        # Singular Value Decomposition for the variance matrix
        # U: (dim, dim), S: (dim,), V: (dim, dim)
        # NOTE: Columns of U are orthonormal vectors
        U, S, V = linalg.svd(var)
        tmp1 = np.dot(U, np.diag(1.0 / np.sqrt(S + eps)))
        tmp2 = np.dot(U, np.diag(np.sqrt(S + eps)))

        zca_mat = np.dot(tmp1, U.T)
        inv_zca_mat = np.dot(tmp2, U.T)

        return {
            'mean': mean,
            'eigen_col_vectors': U,
            'eigen_values': S,
            'zca_mat': zca_mat,
            'inv_zca_mat': inv_zca_mat,
        }

    def fit_and_init_np(self, sess, x):
        assert not self._built, "The model's parameters must have NOT been built!"
        results = self.fit_np(x)

        sess.run([tf.assign(self.eigen_col_vectors, results['eigen_col_vectors']),
                  tf.assign(self.eigen_values, results['eigen_values']),
                  tf.assign(self.zca_mat, results['zca_mat']),
                  tf.assign(self.inv_zca_mat, results['inv_zca_mat']),
                  tf.assign(self.mean, results['mean'])])
        self._built = True

    def transform(self, x, name=None):
        assert self._built, "The model's parameters must have been built!"
        with tf.name_scope(name or "zca_transform"):
            shape = mixed_shape(x)
            z = tf.reshape(tf.matmul(flatten_right_from(x, axis=1) - self.mean, self.zca_mat), shape)
        return z

    def invert(self, x, name=None):
        assert self._built, "The model's parameters must have not been built!"
        with tf.name_scope(name or "zca_invert"):
            shape = mixed_shape(x)
            z = tf.reshape(tf.matmul(flatten_right_from(x, axis=1), self.inv_zca_mat) + self.mean, shape)
        return z
