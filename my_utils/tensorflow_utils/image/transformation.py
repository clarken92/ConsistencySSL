import tensorflow as tf

from ..shaping import mixed_shape


def central_crop_with_fixed_size(image, target_height, target_width):
    image = tf.convert_to_tensor(image, name='image')

    is_batch = True
    ndims = image.get_shape().ndims

    if ndims is None:
        is_batch = False
        image = tf.expand_dims(image, 0)
        image.set_shape([None] * 4)
    elif ndims == 3:
        is_batch = False
        image = tf.expand_dims(image, 0)
    elif ndims != 4:
        raise ValueError('\'image\' must have either 3 or 4 dimensions.')

    image_shape = mixed_shape(image)
    assert len(image_shape) == 4
    input_height, input_width = image_shape[1], image_shape[2]

    offset_height = (input_height - target_height) // 2
    offset_width = (input_width - target_width) // 2

    cropped_image = tf.image.crop_to_bounding_box(
        image, offset_height=offset_height, offset_width=offset_width,
        target_height=target_height, target_width=target_width)

    if not is_batch:
        cropped_image = tf.squeeze(cropped_image, axis=[0])

    return cropped_image


# Use in semi-supervised learning
# From https://github.com/CuriousAI/mean-teacher
# ------------------------------------ #
def random_flip(inputs, horizontally, vertically, is_train, name=None):
    """
    Flip images randomly. Make separate flipping decision for each image.

    inputs (4-D tensor): Input images (batch size, height, width, channels).
    horizontally (bool): If True, flip horizontally with 50% probability. Otherwise, don't.
    vertically (bool): If True, flip vertically with 50% probability. Otherwise, don't.
    is_training (bool): If False, no flip is performed.
    """
    with tf.name_scope(name or "random_flip"):
        batch_size, height, width, _ = tf.unstack(tf.shape(inputs))
        vertical_choices = (tf.random_uniform([batch_size], 0, 2, tf.int32) *
                            tf.to_int32(vertically) * tf.to_int32(is_train))
        horizontal_choices = (tf.random_uniform([batch_size], 0, 2, tf.int32) *
                              tf.to_int32(horizontally) * tf.to_int32(is_train))
        vertically_flipped = tf.reverse_sequence(inputs, vertical_choices * height, 1)
        both_flipped = tf.reverse_sequence(vertically_flipped, horizontal_choices * width, 2)
        return both_flipped


def random_translate(inputs, scale, is_train, padding_mode='REFLECT', name=None):
    """
    Translate images by a random number of pixels
    The dimensions of the image tensor remain the same. Padding is added where necessary, and the
    pixels outside image area are cropped off.
    For performance reasons, the offset values need to be integers and not Tensors.

    inputs (4-D tensor): Input images (batch size, height, width, channels).
    scale (integer): Maximum translation in pixels. For each image on the batch, a random
                     2-D translation is picked uniformly from ([-scale, scale], [-scale, scale]).
    is_train (bool): If False, no translation is performed.
    padding_mode (string): Either 'CONSTANT', 'SYMMETRIC', or 'REFLECT'. What values to use for
                           pixels that are translated from outside the original image.
                           This parameter is passed directly to tensorflow.pad function.
    """
    assert isinstance(scale, int)

    with tf.name_scope(name or "random_translate"):
        def random_offsets(batch_size, minval, inclusive_maxval, name=None):
            with tf.name_scope(name or "random_offsets"):
                return tf.random_uniform(
                    [batch_size], minval=minval, maxval=inclusive_maxval + 1, dtype=tf.int32)

        def do_translate():
            with tf.name_scope("do_translate"):
                batch_size = tf.shape(inputs)[0]
                offset_heights = random_offsets(batch_size, -scale, scale, "offset_heights")
                offset_widths = random_offsets(batch_size, -scale, scale, "offset_widths")
                return translate(inputs, offset_heights, offset_widths, scale, padding_mode)

        return tf.cond(is_train, do_translate, lambda: inputs)


def translate(images, vertical_offsets, horizontal_offsets, scale, padding_mode, name="translate"):
    """
    Translate images
    images (batch, height, width, channels): Input images (batch, height, width, channels).
    vertical_offsets (batch,): Vertical translation in pixels for each image.
    horizontal offsets (batch,): Horizontal translation in pixels.
    scale (integer): Maximum absolute offset (needed for performance reasons).

    padding_mode (string): Either 'CONSTANT', 'SYMMETRIC', or 'REFLECT':
        What values to use for pixels that are translated from outside the original image.
        This parameter is passed directly to 'tensorflow.pad' fuction.
    """
    assert isinstance(scale, int)
    kernel_size = 1 + 2 * scale
    batch_size, height, width, channels = images.get_shape().as_list()

    def assert_shape(tensor, expected_shape):
        tensor_shape = tensor.get_shape().as_list()
        error_message = "tensor {name} shape {actual} != {expected}"
        assert tensor_shape == expected_shape, error_message.format(
            name=tensor.name, actual=tensor_shape, expected=expected_shape)

    def one_hots(offsets, name='one_hots'):
        with tf.name_scope(name) as scope:
            with tf.control_dependencies([tf.assert_less_equal(tf.abs(offsets), scale)]):
                result = tf.expand_dims(tf.one_hot(scale - offsets, kernel_size), 1, name=scope)
                assert_shape(result, [batch_size, 1, kernel_size])
                return result

    def assert_equal_first_dim(tensor_a, tensor_b, name='assert_equal_first_dim'):
        with tf.name_scope(name) as scope:
            first_dims = tf.shape(tensor_a)[0], tf.shape(tensor_b)[0]
            return tf.Assert(tf.equal(*first_dims), first_dims, name=scope)

    with tf.name_scope(name) as scope:
        with tf.control_dependencies([
            assert_equal_first_dim(images, vertical_offsets, "assert_height"),
            assert_equal_first_dim(images, horizontal_offsets, "assert_width")
        ]):
            filters = tf.matmul(one_hots(vertical_offsets),
                                one_hots(horizontal_offsets),
                                adjoint_a=True)
            assert_shape(filters, [batch_size, kernel_size, kernel_size])

            padding_sizes = [[0, 0], [scale, scale], [scale, scale], [0, 0]]
            padded_images = tf.pad(images, padding_sizes, mode=padding_mode)
            assert_shape(padded_images, [batch_size, height + 2 * scale, width + 2 * scale, channels])

            depthwise_inp = tf.transpose(padded_images, perm=[3, 1, 2, 0])
            # assert_shape(depthwise_inp, [channels, height + 2 * scale, width + 2 * scale, batch_size])

            depthwise_filters = tf.expand_dims(tf.transpose(filters, [1, 2, 0]), -1)
            # assert_shape(depthwise_filters, [kernel_size, kernel_size, batch_size, 1])

            convoluted = tf.nn.depthwise_conv2d_native(
                depthwise_inp, depthwise_filters, strides=[1, 1, 1, 1], padding='VALID')
            # assert_shape(convoluted, [channels, height, width, batch_size])

            result = tf.transpose(convoluted, (3, 1, 2, 0), name=scope)
            # assert_shape(result, [batch_size, height, width, channels])

            return result
# ------------------------------------ #