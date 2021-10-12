# -*- coding: utf-8 -*-
# !/usr/bin/python

import os

from src.utils.gpu_util import set_seeds, set_device

set_seeds()
set_device("gpu")

from src.configuration.config import Configuration
from src.computational_graph import training


def main():
    """Main method"""

    config = Configuration().get_configuration()
    training_mode = config["training"]["training_mode"]

    print("Start Training...")
    c_graph = training.Train()

    if config["training"]["training_method"] == "dof":
        c_graph.dof(mode=training_mode)


if __name__ == "__main__":

    main()
