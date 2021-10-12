# -*- coding: utf-8 -*-
# !/usr/bin/python

import os
import sys
import json

from src.utils.general_utils import dump_pickled_data, load_pickled_data, get_filepath, get_folder_path


class Configuration:
    """Configuration reads, validates and return the configuration.json file
    with the parameters for dataset preprocess and training
    """

    def __init__(self):

        self.config = self.create_configuration()

    def get_configuration(self):
        return self.config

    def create_configuration(self):
        """
        Reads .json file and returns a dictionary with configuration
        """
        config = json.load(open(get_filepath("src/configuration", "configuration.json"), "r"))

        return config
