# -*- coding: utf-8 -*-
# !/usr/bin/python

import pandas as pd
import json

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


from src.configuration.config import Configuration
from src.configuration.ml_config import SEED
from src.utils.gpu_util import set_seeds
from src.utils.train_utils import (
    get_optimizer,
    losses_and_metrics,
    model_initialise,
    read_dataset_from_csv,
    dataset_to_tensor,
    active_data_splits,
    plot_training,
)
from src.utils.training_callbacks import early_stopping, model_chpnt, reduce_on_plateau
from src.utils.general_utils import get_artifact_names, dump_pickled_data, get_filepath


def status(opt_name, lr, image_size, k, reps, seed, method, mode, model, split):
    """Print status of configurations

    Arguments:
        opt_name: str, optimizer
        lr: float, learning rate
        image_size: list, image size
    """
    settings = "Model: {}\nOptimizer: {}\nLearning Rate: {}\nImage size: {}\nFilters: {}\nReps: {}".format(
        model, opt_name, lr, image_size, k, reps
    )
    status_filename = get_filepath("trained_models/logs", f"{model}_{method}_{mode}_{split}_{seed}.txt")
    print(settings)

    with open(status_filename, "w") as f:
        f.write(settings)


class Train:
    def __init__(self):

        config = Configuration().get_configuration()

        # hyperparameters
        self.normalize = config["training"]["normalize"]
        self.lr = config["training"]["lr"]
        self.epochs = config["training"]["epochs"]
        self.img_size = [
            config["training"]["img_size_x"],
            config["training"]["img_size_y"],
        ]
        self.batch_size = config["training"]["batch_size"]
        self.num_classes = config["training"]["num_classes"]
        self.cardinality_split = config["al"]["cardinality_split"]
        self.k = config["training"]["filters"]
        self.reps = config["training"]["repetitions"]

        # load dataset paths
        self.dof_dataset_path = config["csv"]["dof"]
        self.model_arch = config["training"]["model_arch"]
        self.al_pool_dataset = config["al"]["pool_dataset"]

        # active learning settings
        self.ceal = config["al"]["ceal"]

        # log status
        status(
            config["training"]["optimizer"],
            config["training"]["lr"],
            self.img_size,
            self.k,
            self.reps,
            SEED,
            config["training"]["training_dataset"],
            config["al"]["training_mode"],
            config["training"]["model_arch"],
            self.cardinality_split,
        )

    def dof(self, mode="baseline"):
        """Active learning process for randonly and actively annotated samples

        Args:
            mode (str, optional): Mode to select sample between baseline/random/active.
        """
        for cardinality_split in self.cardinality_split:

            train_data = read_dataset_from_csv("train", get_filepath("dataset", self.dof_dataset_path))
            valid_data = read_dataset_from_csv("valid", get_filepath("dataset", self.dof_dataset_path))
            test_data = read_dataset_from_csv("test", get_filepath("dataset", self.dof_dataset_path))

            # pool dataset for active learning simulation
            if mode == "random":
                print("Randomly selected samples")
                print(f"------------{cardinality_split}----------")

                pool_dataset = pd.read_csv(self.al_pool_dataset)
                base_dir = "randomly_results"
                (
                    save_model_name,
                    save_training_hist_name,
                    save_training_hist_plot,
                    save_test_results_name,
                    save_cm_name,
                    save_report_name,
                ) = get_artifact_names(base_dir, self.model_arch, cardinality_split, SEED)

                train_data, valid_data = active_data_splits(
                    train_data, valid_data, pool_dataset, cardinality_split, mode=mode
                )

            elif "sqal" in mode:
                base_dir = f"actively_results_{mode}"
                al_inference_results = []
                pseudolabels = 0

                # load active pool
                pool_dataset = pd.read_csv(self.al_pool_dataset)

                # simulate active learning using bootstrap model to inference in the total "unlabelled" instances
                model = load_model(get_filepath("trained_models/baseline", f"{self.model_arch}.h5"), compile=False)

                # get a dataset without labels to inference
                selection_al_ds = dataset_to_tensor(pool_dataset, batch_size=self.batch_size, shuffle=False, mode="al")

                predictions = model.predict(selection_al_ds, workers=1)
                del model

                # create active pool
                for i, (_, img_filename) in enumerate(selection_al_ds.unbatch()):
                    img_name = str(img_filename.numpy().decode("utf8").split("/")[-1])
                    probability = float(predictions[i][np.argmax(predictions[i])])

                    # cost effective al, create pseudolabels
                    if self.ceal and probability >= 0.9:
                        label = int(np.argmax(predictions[i]))
                        pseudolabels += 1

                    # always take the annotated label as groundtruth
                    else:
                        label = pool_dataset.set_index("photo_id").loc[img_name]["label"]

                    al_inference_results.append([img_name, label, probability])

                pool_al_dataset = pd.DataFrame(
                    al_inference_results,
                    columns=["photo_id", "label", "preds"],
                )

                pool_al_dataset_sorted = pool_al_dataset.sort_values(by="preds").reset_index(drop=True)

                # c) train by slices
                print("------------{}----------".format(cardinality_split))

                (
                    save_model_name,
                    save_training_hist_name,
                    save_training_hist_plot,
                    save_test_results_name,
                    save_cm_name,
                    save_report_name,
                ) = get_artifact_names(base_dir, self.model_arch, cardinality_split, SEED)

                # concatenate sliced pool dataset to the baseline dataset
                train_data, valid_data = active_data_splits(
                    train_data, valid_data, pool_al_dataset_sorted, cardinality_split, mode=mode
                )

            elif "all" in mode:
                print("------------{}----------".format(cardinality_split))
                base_dir = f"actively_results_{mode}"

                pseudolabels = 0
                if cardinality_split == 100:
                    # load active pool
                    pool_dataset = pd.read_csv(self.al_pool_dataset)
                    # simulate active learning using bootstrap model to inference in the total "unlabelled" instances
                    model = load_model(get_filepath("trained_models/baseline", f"{self.model_arch}.h5"), compile=False)

                else:
                    previous_selection = pd.read_csv(
                        get_filepath(
                            f"dataset/csv/active_learning/{mode}",
                            f"{self.model_arch}_active_selection_{cardinality_split-100}_{SEED}.csv",
                        )
                    )

                    pool_dataset = pd.read_csv(
                        get_filepath(
                            f"dataset/csv/active_learning/{mode}",
                            f"{self.model_arch}_annotation_pool_{cardinality_split-100}_{SEED}.csv",
                        )
                    )

                    model = load_model(
                        get_filepath(
                            f"trained_models/{base_dir}/{self.model_arch}",
                            f"model_{cardinality_split-100}_seed_{SEED}.h5",
                        ),
                        compile=False,
                    )

                pool_ds = dataset_to_tensor(pool_dataset, batch_size=self.batch_size, shuffle=False, mode="al")
                predictions = model.predict(pool_ds)

                al_inference_results = []

                for i, (_, img_filename) in enumerate(pool_ds.unbatch()):
                    img_name = str(img_filename.numpy().decode("utf8").split("/")[-1])
                    probability = float(predictions[i][np.argmax(predictions[i])])

                    # cost effective al, create pseudolabels
                    if self.ceal and probability >= 0.9:
                        label = int(np.argmax(predictions[i]))
                        pseudolabels += 1

                    # always take the annotated label as groundtruth
                    else:
                        label = pool_dataset.set_index("photo_id").loc[img_name]["label"]

                    al_inference_results.append([img_name, label, probability])

                pool_al_dataset = pd.DataFrame(
                    al_inference_results,
                    columns=["photo_id", "label", "preds"],
                )

                pool_al_dataset_sorted = pool_al_dataset.sort_values(by="preds").reset_index(drop=True)

                # Calculate support bins
                al_inference_results = {}
                for i, (_, img_filename) in enumerate(pool_ds.unbatch()):
                    al_inference_results[str(img_filename.numpy().decode("utf8").split("/")[-1])] = {
                        "class": int(np.argmax(predictions[i])),
                        "prob": float(predictions[i][np.argmax(predictions[i])]),
                    }

                bin_1 = []
                bin_2 = []
                bin_3 = []
                bin_4 = []
                bin_5 = []

                for idx, res in enumerate(al_inference_results.items()):
                    img_name = res[0]
                    pred_class = res[1]["class"]
                    c = res[1]["prob"]
                    if c < 0.6:
                        bin_1.append([img_name, pred_class, c])
                    elif c >= 0.6 and c < 0.7:
                        bin_2.append([img_name, pred_class, c])
                    elif c >= 0.7 and c < 0.8:
                        bin_3.append([img_name, pred_class, c])
                    elif c >= 0.8 and c < 0.9:
                        bin_4.append([img_name, pred_class, c])
                    else:
                        bin_5.append([img_name, pred_class, c])

                support_bins = [len(bin_1), len(bin_2), len(bin_3), len(bin_4), len(bin_5)]

                save_support_bins = get_filepath(
                    f"trained_models/{base_dir}/{self.model_arch}",
                    f"support_bins_active_{cardinality_split}_seed_{SEED}",
                )

                (
                    save_model_name,
                    save_training_hist_name,
                    save_training_hist_plot,
                    save_test_results_name,
                    save_cm_name,
                    save_report_name,
                ) = get_artifact_names(base_dir, self.model_arch, cardinality_split, SEED)

                if cardinality_split == 100:
                    new_active_ds = pool_al_dataset_sorted[:100]
                else:
                    active_selection = pool_al_dataset_sorted[:100]
                    new_active_ds = pd.concat([active_selection, previous_selection])

                # save dataset for next round
                next_pool_dataset = pool_al_dataset_sorted[100:]
                next_pool_dataset.to_csv(
                    get_filepath(
                        f"dataset/csv/active_learning/{mode}",
                        f"{self.model_arch}_annotation_pool_{cardinality_split}_{SEED}.csv",
                    )
                )
                new_active_ds["label"] = new_active_ds["label"].astype(int)
                new_active_ds.to_csv(
                    get_filepath(
                        f"dataset/csv/active_learning/{mode}",
                        f"{self.model_arch}_active_selection_{cardinality_split}_{SEED}.csv",
                    )
                )

                train_data, valid_data = active_data_splits(
                    train_data, valid_data, new_active_ds, cardinality_split, mode="active"
                )

                print("Pseudolabels: {}".format(pseudolabels))
                # save support bins
                dump_pickled_data(save_support_bins, support_bins)

            else:
                base_dir = "baseline"

                save_model_name = get_filepath(f"trained_models/{base_dir}/{self.model_arch}", "model.h5")
                save_training_hist_name = get_filepath(
                    f"trained_models/{base_dir}/{self.model_arch}", "training_hist_model"
                )
                save_test_results_name = get_filepath(
                    f"trained_models/{base_dir}/{self.model_arch}", "test_results_model.json"
                )
                save_cm_name = get_filepath(f"trained_models/{base_dir}/{self.model_arch}", "cm_model")
                save_report_name = get_filepath(f"trained_models/{base_dir}/{self.model_arch}", "report_model")
                save_training_hist_plot = get_filepath(
                    f"trained_models/{base_dir}/{self.model_arch}", "learning_history"
                )

            # start training process with concatenated dataset, baseline dataset + pool dataset(random or active)
            train_ds = dataset_to_tensor(train_data, batch_size=self.batch_size, shuffle=True)
            valid_ds = dataset_to_tensor(valid_data, batch_size=self.batch_size, shuffle=True)
            test_ds = dataset_to_tensor(test_data, batch_size=self.batch_size)

            # Load model
            model = model_initialise()

            # Load callbacks
            callbacks = [
                early_stopping(),
                reduce_on_plateau(),
                model_chpnt(save_model_name),
            ]

            _, _, _, _, _, _, _, _, f1, _ = losses_and_metrics(2)

            optimizer = get_optimizer(self.lr, "adam")
            # Model compile
            model.compile(
                optimizer=optimizer,
                loss="categorical_crossentropy",
                metrics=[f1, "acc"],
            )

            # Model fit
            results = model.fit(
                train_ds,
                validation_data=valid_ds,
                epochs=self.epochs,
                verbose=2,
                callbacks=callbacks,
                workers=0,
                use_multiprocessing=False,
            )

            # save model resutls
            dump_pickled_data(save_training_hist_name, results.history)

            # plot training history
            plot_training(results.history, save_training_hist_plot)

            # inference on test
            model = load_model(save_model_name, compile=True)

            # Evaluate Model
            test_results = model.evaluate(test_ds, workers=1)
            with open(save_test_results_name, mode="w") as f:
                json.dump(dict(zip(model.metrics_names, test_results)), f)

            # Model predictions
            test_preds = model.predict(test_ds, workers=1)

            true_test_y = []
            for _, y in test_ds:
                true_test_y.append(np.argmax(y, axis=1))
                y = np.concatenate(true_test_y)

            dump_pickled_data(save_cm_name, confusion_matrix(y, np.argmax(test_preds, axis=1)))
            dump_pickled_data(
                save_report_name,
                classification_report(y, np.argmax(test_preds, axis=1), digits=3),
            )
            tf.compat.v1.reset_default_graph()
            set_seeds()
