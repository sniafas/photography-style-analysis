from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def early_stopping():
    return EarlyStopping(monitor='val_loss', patience=4)

def model_chpnt():
    return ModelCheckpoint(
        filepath='source/logs/tf_ckpts/fit/cp-{epoch:04d}.ckpt',
        monitor='val_loss',
        verbose=1,
        mode='auto',
        save_best_only=True,
        save_freq='epoch'
        )
