from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def early_stopping():
    return EarlyStopping(monitor="val_loss", patience=6, min_delta=0.001)


def reduce_on_plateau():
    return ReduceLROnPlateau(monitor="val_loss", patience=5, verbose=1)


def model_chpnt(filepath):
    return ModelCheckpoint(
        filepath=filepath,
        monitor="val_loss",
        verbose=1,
        mode="auto",
        save_best_only=True,
    )
