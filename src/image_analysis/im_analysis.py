"""
Module for image analysis and handling methods in unsplash dataset
"""
import pandas as pd
from glob import glob
import os
import cv2
from PIL import Image
from PIL import ImageFile
from sys import argv
from p_tqdm import p_map
import sys

sys.path.insert(0, os.path.abspath("../"))

from configuration.config import Configuration

import matplotlib.pyplot as plt
from tqdm import tqdm

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

    def __init__(self, dataset, dataset_path, write_path):

        assert os.path.exists(dataset), "File {} does not exist".format(dataset)
        assert os.path.exists(dataset_path), "Image path {} does not exist".format(dataset_path)
        assert os.path.exists(write_path), "Write path {} does not exist".format(write_path)
        self.dataset = pd.read_csv(dataset)
        self.dataset_path = dataset_path
        self.write_path = write_path
        self.image_path = glob(dataset_path + "*.jpg")

    def orientation(self):
        """
        Find orientation of an image by comparing the dimensions

        Returns:
            A pandas dataset with orientation column
        """

        for i in range(len(self.image_path)):

            im = Image.open(self.image_path[i])
            im_name = self.image_path[i].split("/")[-1][:-4]
            if im.width > im.height:
                self.dataset.loc[
                    self.dataset.loc[self.dataset["photo_id"] == im_name].index,
                    "orientation",
                ] = 1
            else:
                self.dataset.loc[
                    self.dataset.loc[self.dataset["photo_id"] == im_name].index,
                    "orientation",
                ] = 0

            self.dataset.loc[
                self.dataset.loc[self.dataset["photo_id"] == im_name].index,
                "photo_width",
            ] = im.width
            self.dataset.loc[
                self.dataset.loc[self.dataset["photo_id"] == im_name].index,
                "photo_height",
            ] = im.height
            Image.Image.close(im)

        return self.dataset

    def resize(self, i):
        """
        Downloaded images are on their original size which leads to an unfair amount of data.
        This function reduces and write images to the 1/4 of the original size to save disk space.
        """

        im = Image.open(self.image_path[i])

        # Most of the images are resized successfully
        try:
            im_resized = im.resize((int(im.width / 2), int(im.height / 2)))
            img_name = self.image_path[i].split("/")[-1]
            im_resized.save(self.write_path + img_name)
            # print("Image %s resized" % img_name)

        # When not, due to the very large image or colour encoding, it is handled here
        except:
            print("Cannot be saved")
            print("Retrying...")
            im = im.convert("RGB")
            im_resized = im.resize((int(im.width / 2), int(im.height / 2)), interpolation=cv2.INTER_AREA)
            im_resized.save(self.write_path + img_name)

        # Close stream
        Image.Image.close(im)

    def resize2ratio(self, orientation, target_width, target_height, verbose=False):
        """
        Resize and write images to disk, according to the specified ratio. Default set is 3:2
        Reminder: Opencv defines image(y,x)

        Args:
            orientation: filter orientation in dataset, 0: vertical, 1:horizontal, -1: read all images
            target_width: target image width
            target_height: target image height
            verbose: show some images
        """

        # Filter orientation - find image ids
        if orientation == -1:
            image_ids = self.dataset["photo_id"]

        else:
            image_ids = self.dataset["photo_id"].loc[self.dataset["orientation"] == orientation].values

        # Make image paths
        image_paths = self.dataset_path + image_ids  # + '.jpg'
        # image_paths = ['/media/steve/Data2/unsplash-dataset/unsplash-original/pjP9usKbIbM.jpg']

        # Make output path
        write_ids = self.write_path + image_ids  # + '.jpg'
        # write_ids = ['/home/steve/Documents/Msc data science/Thesis/src/dataset/images/pjP9usKbIbM.jpg']

        # Fix target image ratio
        self.target_ratio_width = target_width
        self.target_ratio_height = target_height

        for i, im_name in enumerate(tqdm(image_paths)):

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
                if new_image.shape[0] != self.target_ratio_height:
                    new_image = self.fix_ratio_y(new_image, self.target_ratio_height, 0)

                # Horizontal images - Fix ratio across X axis
                if new_image.shape[1] != self.target_ratio_width:
                    new_image = self.fix_ratio_x(new_image, self.target_ratio_width, 1)

            else:

                if new_image.shape[0] != self.target_ratio_height:
                    new_image = self.fix_ratio_x(new_image, self.target_ratio_height, 0)

                # Vertical images - Fix ratio across X axis
                if new_image.shape[1] != self.target_ratio_width:
                    new_image = self.fix_ratio_y(new_image, self.target_ratio_width, 1)

            if verbose:
                plt.figure(1)
                plt.axis("off")
                plt.imshow(new_image)

            if orientation == 1:

                if new_image.shape[0] != self.target_ratio_height or new_image.shape[1] != self.target_ratio_width:
                    print("Problem %s" % write_ids[i])
                else:
                    cv2.imwrite(write_ids[i], new_image)
            else:
                if new_image.shape[0] != self.target_ratio_height or new_image.shape[1] != self.target_ratio_width:
                    print("Problem %s" % write_ids[i])
                    print(new_image.shape[0])
                    print(new_image.shape[1])
                else:
                    cv2.imwrite(write_ids[i], new_image)

    def fix_ratio_x(self, img, target_ratio_width, axis):
        """
        Crops and/or pads a vertical image to a target width.
        Given target ratio (width of the image) is used as a rule of thumb to transform
        x axis (width of the image) and follow the given aspect ratio
        If `width` is greater than the specified `target_ratio_width` image is evenly cropped along that dimension.
        If `width` is smaller than the specified `target_ratio_width` image is evenly padded with 0 along that dimension.

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
        If `height` is smaller than the specified `target_ratio_height` image is evenly padded with 0 along that dimension.

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

    def main(self):

        # Run image resize
        # p_map(self.resize, list(range(0, len(self.image_path))), num_cpus=8)
        # sEDzxW4NhL4 has issue

        self.resize2ratio(0, 200, 300)


if __name__ == "__main__":

    config = Configuration().get_configuration()

    dataset = config["dataset"]["dataset_path"]
    input_image_path = config["dataset"]["img_path"]
    write_path = config["dataset"]["resized_path"]

    # ia = ImageAnalysis(
    #    dataset='../../../dataset/horizontal/train_horizontal.csv', dataset_path='/media/steve/Data2/unsplash-dataset/unsplashed-resized/', write_path='/media/steve/Data2/unsplash-dataset/unsplashed-horizontal/')
    ia = ImageAnalysis(dataset=dataset, dataset_path=input_image_path, write_path=write_path)

    #'/media/steve/Data2/unsplash-dataset/unsplash-v1.1.0+/' -> resize dataset_path
    #'/media/steve/Data2/unsplash-dataset/unsplashed-resized/' -> resize write_path
    ia.main()
