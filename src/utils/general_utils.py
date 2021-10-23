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


def get_artifact_names(base_dir: str, model_arch: str, cardinality_split: int, seed: int):
    """Create artifact save names

    Args:
        base_dir (str): base dir run name
        model_arch (str): model arch
        cardinality_split (int): number of samples appended
        seed (int): seed
    """

    save_model_name = get_filepath(
        f"trained_models/{base_dir}/{model_arch}", f"model_{cardinality_split}_seed_{seed}.h5"
    )
    save_training_hist_name = get_filepath(
        f"trained_models/{base_dir}/{model_arch}",
        f"training_hist_model_{cardinality_split}_seed_{seed}",
    )
    save_training_hist_plot = get_filepath(
        f"trained_models/{base_dir}/{model_arch}",
        f"learning_history_{cardinality_split}_seed_{seed}",
    )
    save_test_results_name = get_filepath(
        f"trained_models/{base_dir}/{model_arch}",
        f"test_results_model_{cardinality_split}_seed_{seed}.json",
    )
    save_cm_name = get_filepath(f"trained_models/{base_dir}/{model_arch}", f"cm_model_{cardinality_split}_{seed}")
    save_report_name = get_filepath(
        f"trained_models/{base_dir}/{model_arch}", f"report_model_{cardinality_split}_seed_{seed}"
    )

    return (
        save_model_name,
        save_training_hist_name,
        save_training_hist_plot,
        save_test_results_name,
        save_cm_name,
        save_report_name,
    )
