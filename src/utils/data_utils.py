import os
from glob import glob
import pandas as pd
import pickle
import numpy as np


def clean_exposures(photos, c):
    """
    Exposure values contain 's' and 'second' in some long exposure captures and spaces as well
    """
    filtered = photos.exif_exposure_time.str.contains(c)
    # Find indices and locate them
    idx = photos["exif_exposure_time"].iloc[photos.loc[filtered == True].index].index
    for i in idx:
        a = photos["exif_exposure_time"].iloc[i].split(c)[0].replace("Second", "").replace(" ", "")

        print("Raw: " + photos["exif_exposure_time"].iloc[i])
        photos["exif_exposure_time"].iloc[i] = a
        print(photos["exif_exposure_time"].iloc[i])
        print(" ")


def clean_focal(photos, c):
    # TODO check for 0 values -> NaN
    """
    There are values containing mm, - and ~ and spaces as well
    """
    filtered = photos.exif_focal_length.str.contains(c)
    # Find indices and locate them
    idx = photos["exif_focal_length"].iloc[photos.loc[filtered == True].index].index
    delimiters = c.split("|")
    for i in idx:
        print("Raw: " + photos["exif_focal_length"].iloc[i])

        a = photos["exif_focal_length"].iloc[i].replace(" ", "").replace(delimiters[0], ".0")

        if delimiters[1] in a:
            photos["exif_focal_length"].iloc[i] = np.NaN
            print(a + "->NaN")
        elif delimiters[2] in a:
            photos["exif_focal_length"].iloc[i] = np.NaN
            print(a + "_>NaN")
        elif a == 0:
            photos["exif_focal_length"].iloc[i] = np.NaN
        else:
            photos["exif_focal_length"].iloc[i] = a

        print(photos["exif_focal_length"].iloc[i])

    flt = np.where(photos.exif_focal_length.str.contains("\."), True, False)
    idx = photos["exif_focal_length"].iloc[photos.loc[flt == False].index].index
    photos["exif_focal_length"].iloc[idx] += ".0"


def clean_iso(photos):
    """
    Find indices that have ISO 0.0 and convert to NaN
    """
    idx = photos["exif_iso"].loc[photos["exif_iso"] == 0.0].index
    for i in idx:
        photos["exif_iso"].iloc[i] = np.NaN


def clean_apertures(photos, c):
    """
    Aperture values are tricky, there are inf, Inf, undef along with f/, f
    Those values have to be replaced with NaN before deal integer like string values
    We have to drop them first, as the complementary .0 will not work with NaNs
    """
    filtered = photos.exif_aperture_value.str.contains(c)

    idx = photos["exif_aperture_value"].iloc[photos.loc[filtered == True].index].index
    delimiters = c.split("|")
    for i in idx:
        print("Raw: " + photos["exif_aperture_value"].iloc[i])
        a = photos["exif_aperture_value"].iloc[i]
        if "inf" in a:
            photos["exif_aperture_value"].iloc[i] = np.NaN
        elif "Inf" in a:
            photos["exif_aperture_value"].iloc[i] = np.NaN
        elif "undef" in a:
            photos["exif_aperture_value"].iloc[i] = np.NaN
        else:
            photos["exif_aperture_value"].iloc[i] = (
                a.replace(delimiters[0], ".").replace(delimiters[1], "").replace(delimiters[5], "")
            )

    flt = np.where(photos.exif_aperture_value.str.contains("\."), True, False)
    idx = photos["exif_aperture_value"].iloc[photos.loc[flt == False].index].index
    photos["exif_aperture_value"].iloc[idx] += ".0"


def process_new_dataset(dataset_path, img_path, write_path):
    """
    Use this function when a new unslash dataset is released.
    Usually in new versions, some images are replaced by new ones due to their platform updates.
    This function reads the new dataset, and detects the discrepancies against the already downloaded images. Then it filters out the new images and runs a data cleaning procedure.
    New_photos.csv is saved and
     1) can be loaded in exploratory analysis to get merged with the previous dataset
     2) can be used as input in download.py to fetch the new images
    Args:
        dataset_path: Unsplash .tsv path
        img_path: Image path of downloaded images
        write_path: Path to write new_photo.csv
    """

    documents = ["photos"]
    datasets = {}
    for doc in documents:
        files = glob(dataset_path + doc + ".tsv*")
        subsets = []
        for filename in files:
            df = pd.read_csv(filename, sep="\t", header=0)
            subsets.append(df)
        datasets[doc] = pd.concat(subsets, axis=0, ignore_index=True)

    df = datasets.copy()
    photos = df["photos"].loc[
        :,
        [
            "photo_id",
            "photo_url",
            "photo_image_url",
            "exif_camera_make",
            "exif_camera_model",
            "exif_iso",
            "exif_focal_length",
            "exif_aperture_value",
            "exif_exposure_time",
            "photo_width",
            "photo_height",
            "photo_aspect_ratio",
        ],
    ]

    # Find discrepancies in the new dataset
    remained_ids = []
    for p in photos["photo_id"]:
        if not os.path.exists(img_path + p + ".jpg"):
            remained_ids.append(photos.loc[photos["photo_id"] == p].index.values)

    # Filter out new photos
    new_photos = photos.take(np.array(remained_ids).ravel())
    print(new_photos.shape)

    # Reset index
    new_photos.reset_index(drop=True, inplace=True)

    # Cleanup procedure
    clean_iso(new_photos)
    clean_focal(new_photos, "mm|-|~")
    clean_exposures(new_photos, "s")
    clean_apertures(new_photos, ",|f/|undef|inf|Inf|f")

    new_photos.to_csv(write_path + "new_photos.csv")


def train_valid_test_split(dataset):
    """
    Split training, validation, test sets 80/10/10
    Arguments:
        dataset: pd.Dataframe

    Returns:
        train, valid, test, test_nan: pd.Dataframe
    """

    train_valid_test_dataset = dataset.dropna(
        axis=0,
        inplace=False,
        subset=[
            "exif_iso",
            "iso_noise_bin_label",
            "exif_focal_length",
            "exif_exposure_time",
            "exif_aperture_value",
            "orientation",
            "iso_noise_label",
            "DoF",
            "DoF_bin",
            "exposure_label",
            "focal_label",
            "focal_label_bin",
        ],
    )

    # test with NaN = full - not null indices
    test_nan = dataset.drop(index=train_valid_test_dataset.index)

    # Train/valid/test -> 80/10/10
    holdout_length = int(train_valid_test_dataset.shape[0] * 20 / 100)
    limit = train_valid_test_dataset.shape[0] - holdout_length
    valid_test_len = int((train_valid_test_dataset.shape[0] - limit) / 2)

    train = train_valid_test_dataset[0:limit]
    valid = train_valid_test_dataset[limit : limit + valid_test_len]
    test = train_valid_test_dataset[limit + valid_test_len :]

    return train, valid, test, test_nan
