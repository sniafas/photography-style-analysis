# -*- coding: utf-8 -*-
# !/usr/bin/python

import os
import pickle


def load_pickled_data(path: str) -> None:

    with open(path, "rb") as f:
        return pickle.load(f)


def dump_pickled_data(path: str, data: object) -> None:

    with open(path, "wb") as f:
        pickle.dump(data, f)


def get_folder_path(path_from_module: str) -> str:

    fn = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))).split("src")[0]
    return "{0}{1}".format(fn, path_from_module)


def get_filepath(path_from_module: str, file_name: str) -> str:

    fn = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))).split("src")[0]
    return "{0}{1}/{2}".format(fn, path_from_module, file_name)
