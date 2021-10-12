import tensorflow as tf


def normalization(x, y, filename):
    # def normalization(x, y):
    # x = x/.255
    x = tf.image.per_image_standardization(x)

    return x, y, filename


def decode_img(img):
    """Decode jpg to tensor

    Args:
        img ([type]): [description]

    Returns:
        [type]: [description]
    """
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.per_image_standardization(img)

    return img


def oh_label(label):
    # Integer encode the label
    return tf.one_hot(label, depth=2)
