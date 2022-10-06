import numpy as np
import tensorflow as tf

def train_with_cross_validate(dataset, model, folds_num, config):
    len_fold = int(len(dataset['X_train']) / folds_num)
    for fold in range(folds_num):
        X_train = np.vstack((dataset['X_train'][0:fold*len_fold], dataset['X_train'][(fold+1)*len_fold:]))
        X_val = dataset['X_train'][fold*len_fold:(fold+1)*len_fold]

        y_train = np.array(dataset['y_train'][0:fold*len_fold] + dataset['y_train'][(fold+1)*len_fold:])
        y_val = np.array(dataset['y_train'][fold*len_fold:(fold+1)*len_fold])

        history = model.model.fit(
            X_train, 
            y_train, 
            epochs=config['train']['epoch'], 
            validation_data=(X_val, y_val),
            batch_size=config['train']['batch_size'],
            callbacks=[tf.keras.callbacks.EarlyStopping(
                patience=5, 
                restore_best_weights=True,
            ), tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                patience=2
            )]
        )
    
    return history