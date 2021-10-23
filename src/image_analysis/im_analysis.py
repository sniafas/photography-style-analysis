# -*- coding: utf-8 -*-
# !/usr/bin/python

import os
import cv2
import pandas as pd
import sys
import matplotlib.pyplot as plt

from glob import glob
from PIL import Image, ImageFile
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("../../"))
from src.configuration.config import Configuration
from src.utils.general_utils import get_filepath, get_folder_path


# Use them for resize
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageAnalysis(object):
    """
    This class image analysis tasks for dataset preparation

    Args:
        dataset: String path of .csv file
        image_path: String path of images to load
        write_path: String path of images to write
    """

    def __init__(self, dataset_path, write_path):

        dataset_path = get_folder_path(f"{dataset_path}")
        write_path = get_folder_path(f"{write_path}")

        assert os.path.exists(dataset_path), "Image path {} does not exist".format(dataset_path)
        assert os.path.exists(write_path), "Write path {} does not exist".format(write_path)

        self.dataset_path = dataset_path
        self.write_path = write_path

    def resize2ratio(self, orientation, target_width, target_height, verbose=False):
        """
        Resize and write images to disk, according to the specified ratio. Default set is 3:2
        Reminder: Opencv defines image(y,x)

        Args:
            orientation: filter orientation in dataset, 0: vertical, 1:horizontal
            target_width: target image width
            target_height: target image height
            verbose: show some images
        """

        # Make image paths
        image_paths = glob(self.dataset_path + "*.jpg")
        print(image_paths)

        for i, im_name in enumerate(tqdm(image_paths)):

            target_path = f"{self.write_path}/{im_name.split('/')[-1:][0]}"
            # print("Opening... %s" % im_name)
            im = cv2.imread(im_name)
            old_size = im.shape[:2]

            if orientation == 1:  # horizontal
                ratio = float(target_width) / max(old_size)
            else:  # vertical
                ratio = float(target_height) / max(old_size)

            # Scale image dimensions according to the ratio
            new_size = tuple(int(x * ratio) for x in old_size)

            if verbose:
                new_image = cv2.cvtColor(
                    cv2.resize(im, (new_size[1], new_size[0]), interpolation=cv2.INTER_AREA),
                    cv2.COLOR_BGR2RGB,
                )
                plt.figure(0)
                plt.axis("off")
                plt.imshow(new_image)
            else:
                new_image = cv2.resize(im, (new_size[1], new_size[0]), interpolation=cv2.INTER_AREA)

            # Horizontal images - Fix ratio across Y axis
            if orientation == 1:
                if new_image.shape[0] != target_height:
                    new_image = self.fix_ratio_y(new_image, target_height, 0)

                # Horizontal images - Fix ratio across X axis
                if new_image.shape[1] != target_width:
                    new_image = self.fix_ratio_x(new_image, target_width, 1)

            else:

                if new_image.shape[0] != target_height:
                    new_image = self.fix_ratio_x(new_image, target_height, 0)

                # Vertical images - Fix ratio across X axis
                if new_image.shape[1] != target_width:
                    new_image = self.fix_ratio_y(new_image, target_width, 1)

            if verbose:
                plt.figure(1)
                plt.axis("off")
                plt.imshow(new_image)

            if orientation == 1:

                if new_image.shape[0] != target_height or new_image.shape[1] != target_width:
                    print("Problem %s" % target_path)
                else:
                    cv2.imwrite(target_path, new_image)
            else:
                if new_image.shape[0] != target_height or new_image.shape[1] != target_width:
                    print("Problem %s" % target_path)
                    print(new_image.shape[0])
                    print(new_image.shape[1])
                else:
                    cv2.imwrite(target_path, new_image)

    def fix_ratio_x(self, img, target_ratio_width, axis):
        """
        Crops and/or pads a vertical image to a target width.
        Given target ratio (width of the image) is used as a rule of thumb to transform
        x axis (width of the image) and follow the given aspect ratio
        If `width` is greater than the specified `target_ratio_width` image is evenly cropped along that dimension.
        If `width` is smaller than the specified `target_ratio_width` image is evenly padded with 0
        along that dimension.

        Args:
            img: cv2 read image
            target_ratio_width: Integer of the target width

        Returns:
            img: cv2 image
        """

        # Crop pixels from image
        if img.shape[axis] > target_ratio_width:

            # Find how many width rows to crop
            rows_to_crop = img.shape[axis] - target_ratio_width

            # Check if rows_to_crop is even or odd, in case of odd remove one more line from the start
            split_rows = int(rows_to_crop / 2)
            if rows_to_crop % 2 == 0:
                img = img[img.shape[axis] - split_rows : split_rows]
            else:
                img = img[img.shape[axis] - split_rows : split_rows + 1]

        # 0Pad pixels to image
        elif img.shape[axis] < target_ratio_width:

            # Find how many width rows to crop
            rows_to_pad = target_ratio_width - img.shape[axis]

            # Check if rows_to_pad is even or odd, in case of odd remove one more line from the start
            split_rows = int(rows_to_pad / 2)
            split_rows + 1 if split_rows == 0 else split_rows
            # Check if rows_to_pad is even or odd, in case of odd, 0pad one more line from the start
            if rows_to_pad % 2 == 0:

                img = cv2.copyMakeBorder(
                    img,
                    split_rows,
                    split_rows,
                    0,
                    0,
                    cv2.BORDER_CONSTANT,
                    value=[0, 0, 0],
                )
            else:
                img = cv2.copyMakeBorder(
                    img,
                    split_rows,
                    split_rows + 1,
                    0,
                    0,
                    cv2.BORDER_CONSTANT,
                    value=[0, 0, 0],
                )

        return img

    def fix_ratio_y(self, img, target_ratio_height, axis):
        """
        Crops and/or pads a horizontal image to a target height.
        Given target ratio (height of the image) is used as a rule of thumb to transform
        y axis (height of the image) and follow the given aspect ratio
        If `height` is greater than the specified `target_ratio_height` image is evenly cropped along that dimension.
        If `height` is smaller than the specified `target_ratio_height` image is evenly padded with 0
        along that dimension.

        Args:
            img: cv2 image
            target_ratio_width: Integer of the target width

        Returns:
            img: cv2 image
        """
        if axis == 0:
            if img.shape[0] > target_ratio_height:

                # Find how many width rows to crop
                rows_to_crop = img.shape[0] - target_ratio_height

                # Divide rows to apply the operation evenly.
                split_rows = int(rows_to_crop / 2)

                # Check if rows_to_crop is even or odd, in case of odd remove one more line from the start
                if rows_to_crop % 2 == 0:
                    img = img[split_rows : img.shape[0] - split_rows]
                else:
                    img = img[split_rows + 1 : img.shape[0] - split_rows]

                # count_cropped += 1

            elif img.shape[0] < target_ratio_height:

                rows_to_pad = target_ratio_height - img.shape[0]  # Calculate pixels to 0pad

                # Share 0pad pixels
                split_rows = int(rows_to_pad / 2)

                # Check if rows_to_pad is even or odd, in case of odd, 0pad one more line from the start
                if rows_to_pad % 2 == 0:
                    img = cv2.copyMakeBorder(
                        img,
                        split_rows,
                        split_rows,
                        0,
                        0,
                        cv2.BORDER_CONSTANT,
                        value=[0, 0, 0],
                    )
                else:
                    img = cv2.copyMakeBorder(
                        img,
                        split_rows + 1,
                        split_rows,
                        0,
                        0,
                        cv2.BORDER_CONSTANT,
                        value=[0, 0, 0],
                    )

        else:
            if img.shape[1] > target_ratio_height:

                # Find how many width rows to crop
                rows_to_crop = img.shape[1] - target_ratio_height

                # Divide rows to apply the operation evenly.
                split_rows = int(rows_to_crop / 2)

                # Check if rows_to_crop is even or odd, in case of odd remove one more line from the start
                if rows_to_crop % 2 == 0:
                    img = img[:, split_rows : img.shape[1] - split_rows]
                else:
                    img = img[:, split_rows + 1 : img.shape[1] - split_rows]

                # count_cropped += 1

            elif img.shape[1] < target_ratio_height:

                rows_to_pad = target_ratio_height - img.shape[1]  # Calculate pixels to 0pad

                # Share 0pad pixels
                split_rows = int(rows_to_pad / 2)

                # Check if rows_to_pad is even or odd, in case of odd, 0pad one more line from the start
                if rows_to_pad % 2 == 0:
                    img = cv2.copyMakeBorder(
                        img,
                        0,
                        0,
                        split_rows,
                        split_rows,
                        cv2.BORDER_CONSTANT,
                        value=[0, 0, 0],
                    )
                else:
                    img = cv2.copyMakeBorder(
                        img,
                        0,
                        0,
                        split_rows + 1,
                        split_rows,
                        cv2.BORDER_CONSTANT,
                        value=[0, 0, 0],
                    )

        return img


if __name__ == "__main__":

    config = Configuration().get_configuration()

    dataset = config["dataset"]["image_transform_dataset_path"]
    input_image_path = config["dataset"]["img_path"]
    write_path = config["dataset"]["resized_path"]

    ia = ImageAnalysis(dataset_path=input_image_path, write_path=write_path)

    # orientation, width, height
    ia.resize2ratio(1, 300, 200)
