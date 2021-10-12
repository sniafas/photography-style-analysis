import tensorflow as tf


def normalization(x, y):
    x = tf.image.per_image_standardization(x)
    return x, y


def augment_with_flip(image, label):

    # image, label = normalize(image, label)
    # Add 6 pixels of padding
    # image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)
    # Random crop back to the original size
    # image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3])
    image = tf.image.random_flip_up_down(image)

    image = tf.image.random_brightness(image, max_delta=0.5)  # Random brightness
    # image = tf.clip_by_value(image, 0, 1)

    return image, label


def augment_with_rotate(x, y):
    """
    Augmentation operations
    """

    # Rotate counterclockwise by 90 degrees num_rot times.
    x = tf.image.rot90(x, k=3)

    # Randomly flip the image vertically with prob 0.5.
    x = tf.image.random_flip_up_down(x)

    # Randomly flip the image horizontally with prob 0.5.
    x = tf.image.random_flip_left_right(x)

    return x, y


def augment_patches(x, y):
    """
    Patch extraction augmentation
    """
    image = tf.expand_dims(x, 0)
    x = tf.image.extract_patches(
        images=image, sizes=[1, 200, 300, 1], strides=[1, 200, 300, 1], rates=[1, 1, 1, 1], padding="VALID"
    )

    return x, y


def patches_random(x, y):

    x = tf.image.random_crop(x, size=[200, 300, 3])

    return x, y


"""
class Augmentation:

    def __init__(self, data_type):

        self.data_type = data_type

    def __call__(self, x, y):
        print("Data have been augmented")
        return self.augment_map(x,y)

"""
