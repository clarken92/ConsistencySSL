from my_utils.tensorflow_utils.layers import BaseLayer
from my_utils.tensorflow_utils.image import random_translate, random_flip, ZCA, Standardization
from my_utils.tensorflow_utils.nn.noise import gauss_noise


class InputPerturber(BaseLayer):
    def __init__(self,
                 normalizer=None,
                 flip_horizontally=True, flip_vertically=False,
                 translating_pixels=2, noise_std=0.15, scope=None):
        BaseLayer.__init__(self, scope)

        if normalizer is not None:
            assert isinstance(normalizer, (ZCA, Standardization)), \
                "'normalizer' must be either ZCA or Standardization!"
        self.normalizer = normalizer

        self.flip_horizontally = flip_horizontally
        self.flip_vertically = flip_vertically
        self.translating_pixels = translating_pixels
        self.noise_std = noise_std

        # print("normalizer: {}".format(self.normalizer))
        # print("flip_horizontally: {}".format(self.flip_horizontally))
        # print("flip_vertically: {}".format(self.flip_vertically))
        # print("translating_pixels: {}".format(self.translating_pixels))
        # print("noise_std: {}".format(self.noise_std))

    def __call__(self, x, is_train):
        if self.normalizer is not None:
            x = self.normalizer.transform(x)
        if self.flip_horizontally or self.flip_vertically:
            x = random_flip(x, horizontally=self.flip_horizontally,
                            vertically=self.flip_vertically, is_train=is_train)
        if self.translating_pixels > 0:
            x = random_translate(x, scale=self.translating_pixels, is_train=is_train)
        if self.noise_std > 0.0:
            x = gauss_noise(x, is_train=is_train, std=self.noise_std)

        return x
