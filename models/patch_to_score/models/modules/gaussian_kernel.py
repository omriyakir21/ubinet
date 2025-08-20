from typing import Tuple
import tensorflow as tf
from keras.constraints import NonNeg
import numpy as np
from sklearn.mixture import GaussianMixture
import keras
from keras.layers import Layer
from keras.constraints import Constraint
from keras.initializers import Initializer


class Init2Value(Initializer):
    def __init__(self, value):
        self.value = value

    def __call__(self, shape, dtype=None):
        return self.value.astype(np.float32)


class ConstraintBetween(Constraint):
    def __init__(self, minimum=-1, maximum=+1):
        self.minimum = minimum
        self.maximum = maximum

    def __call__(self, w):
        return tf.clip(w, self.minimum, self.maximum)


class GaussianKernel(Layer):
    def __init__(self, N, initial_values, covariance_type='diag', eps=1e-1, **kwargs):
        super(GaussianKernel, self).__init__(**kwargs)
        self.supports_masking = True
        self.eps = eps
        self.N = N
        self.initial_values = initial_values
        self.covariance_type = covariance_type
        assert self.covariance_type in ['diag', 'full']

    def build(self, input_shape):
        self.nbatch_dim = len(input_shape) - 1
        self.d = input_shape[-1]

        self.center_shape = [self.d, self.N]

        self.centers = self.add_weight(shape=self.center_shape, name='centers',
                                       initializer=Init2Value(
                                           self.initial_values[0]),
                                       regularizer=None,
                                       constraint=None)

        if self.covariance_type == 'diag':
            self.width_shape = [self.d, self.N]

            self.widths = self.add_weight(shape=self.width_shape,
                                          name='widths',
                                          initializer=Init2Value(
                                              self.initial_values[1]),
                                          regularizer=None,
                                          constraint=NonNeg())

        elif self.covariance_type == 'full':
            self.sqrt_precision_shape = [self.d, self.d, self.N]

            self.sqrt_precision = self.add_weight(shape=self.sqrt_precision_shape,
                                                  name='sqrt_precision',
                                                  initializer=Init2Value(
                                                      self.initial_values[1]),
                                                  regularizer=None,
                                                  constraint=ConstraintBetween(-1/self.eps, 1/self.eps))

        super(GaussianKernel, self).build(input_shape)

    def call(self, inputs, mask=None):
        if self.covariance_type == 'diag':
            activity = tf.exp(- 0.5 * tf.math.reduce_sum(
                (
                    (
                        tf.expand_dims(inputs, axis=-1)
                        - tf.reshape(self.centers,
                                     [1 for _ in range(self.nbatch_dim)] + self.center_shape)
                    ) / tf.reshape(self.eps + self.widths, [1 for _ in range(self.nbatch_dim)] + self.width_shape)
                )**2, axis=-2))

        elif self.covariance_type == 'full':
            intermediate2 = tf.einsum('...i,lij->...lj', inputs, self.sqrt_precision) - tf.reshape(tf.einsum(
                'ij,lij->lj', self.centers, self.sqrt_precision), [1 for _ in range(self.nbatch_dim)] + self.center_shape)
            activity = tf.exp(- 0.5 *
                              tf.math.reduce_sum(intermediate2**2, axis=-2))

        # zero out masked values
        return activity * tf.cast(mask[..., None], activity.dtype)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape[:-1]) + [self.N]
        return tuple(output_shape)

    def get_config(self):
        config = {'N': self.N,
                  'initial_values': self.initial_values,
                  'covariance_type': self.covariance_type}
        base_config = super(
            GaussianKernel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    # TODO: fix model loading
    # @classmethod
    # def from_config(cls, config):
        # initial_values_config = config.pop('initial_values')
        # initial_values = keras.saving.deserialize_keras_object(
        #     initial_values_config)
        # return cls(initial_values=initial_values, **config)
        # return cls(**config)


def inv_root_matrix(H):
    lam, v = np.linalg.eigh(H)
    return np.dot(v,  1 / np.sqrt(lam)[:, np.newaxis] * v.T)


def initialize_GaussianKernel(points, N, covariance_type='diag', reg_covar=1e-1, n_init=10):
    GMM = GaussianMixture(n_components=N, covariance_type=covariance_type, verbose=1,
                          reg_covar=reg_covar, n_init=n_init)
    GMM.fit(points)
    centers = GMM.means_
    covariances = GMM.covariances_
    probas = GMM.weights_
    order = np.argsort(probas)[::-1]
    centers = centers[order]
    covariances = covariances[order]
    probas = probas[order]
    if covariance_type == 'diag':
        widths = np.sqrt(covariances)
    elif covariance_type == 'full':
        sqrt_precision_matrix = np.array(
            [inv_root_matrix(covariance) for covariance in covariances])
    if covariance_type == 'diag':
        return centers.T, widths.T
    elif covariance_type == 'full':
        return centers.T, sqrt_precision_matrix.T


def initialize_GaussianKernelRandom(xlims, N, covariance_type):
    xlims = np.array(xlims, dtype=np.float32)
    coordinates_dimension = xlims.shape[0]

    centers = np.random.rand(coordinates_dimension, N).astype(np.float32)
    centers = centers * (xlims[:, 1]-xlims[:, 0])[:,
                                                  np.newaxis] + xlims[:, 0][:, np.newaxis]

    widths = np.ones([coordinates_dimension, N], dtype=np.float32)
    widths = widths * (xlims[:, 1] - xlims[:, 0])[:, np.newaxis] / (N / 4)

    if covariance_type == 'diag':
        initial_values = [centers, widths]
    else:
        sqrt_precision_matrix = np.stack(
            [np.diag(1.0/(1e-4+widths[:, n])).astype(np.float32) for n in range(N)], axis=-1)
        initial_values = [centers, sqrt_precision_matrix]
    return initial_values


def initialize_gaussian_kernel_uniform(xrange: Tuple[float, float], N: int) -> GaussianKernel:
    centers = np.linspace(xrange[0], xrange[1], N)
    widths = [((xrange[1] - xrange[0]) / (N / 4)) for _ in range(N)]
    initial_values = [np.array([centers]), np.array([widths])]
    kernel = GaussianKernel(N, initial_values, 'diag')
    return kernel
