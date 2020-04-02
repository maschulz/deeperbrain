import logging
import os
import uuid

import keras
import numpy
from keras import backend as K
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Reshape, \
    GlobalAveragePooling2D
from keras.models import Sequential
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, accuracy_score
from sklearn.model_selection import ParameterGrid

PATH = '.'


def batch_generator(x, y, batch_size=32):
    while 1:
        idx = numpy.random.permutation(len(x))
        for i in range(len(x) // batch_size):
            batch_idx = idx[i * batch_size:(i + 1) * batch_size]
            yield x[batch_idx], y[batch_idx]


class Base_Model:
    def __init__(self, early_stopping=10, lr_factor=0.5, lr_patience=3, batch_size=32, steps_per_epoch=500, epochs=160,
                 name=None, multiclass_method='xent'):
        self.early_stopping = early_stopping
        self.epochs = epochs
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.name = name
        self.multiclass_method = multiclass_method

    # temporary version w/o keeping weights
    def run_grid(self, x, y, train_idx, val_idx, test_idx, grid, hyperparameters):
        self.shape = x.shape[1:]
        self.n_classes = len(numpy.unique(y))

        if self.multiclass_method == 'xent':
            self.n_outputs = self.n_classes
        elif self.multiclass_method == 'ovr':
            self.n_outputs = 2

        for hp_key in ParameterGrid(grid):
            val_score = hyperparameters.get(hp_key)

            if val_score is None:
                K.clear_session()  # this is important - graphs accumulate...
                K.set_session(
                    K.tf.Session(
                        config=K.tf.ConfigProto(intra_op_parallelism_threads=1,
                                                inter_op_parallelism_threads=1)))

                path, epochs, final_xent = self.fit(hp_key, x[train_idx], y[train_idx], x[val_idx], y[val_idx])
                train_score = self.score(path, x[train_idx], y[train_idx], 'train')
                val_score = self.score(path, x[val_idx], y[val_idx], 'val')
                test_score = self.score(path, x[test_idx], y[test_idx], 'test')
                self.delete_weights(path)

                hp_result = {**hp_key, **train_score, **val_score, **test_score,
                             **{'path': path, 'epochs': epochs, 'final_xent': final_xent}}

                hyperparameters.set(hp_result)
                logging.debug(f'hp_trial {hp_result}')

        hp_key = hyperparameters.best()
        return hp_key

    def get_model(self, **args):
        raise NotImplementedError

    def fit_model(self, hp_key, x, y, x_val, y_val, path):
        model = self.get_model(**hp_key)
        save_best = ModelCheckpoint(f'{PATH}/nn_weights/{path}.w', monitor='val_loss', verbose=1,
                                    save_best_only=True,
                                    mode='auto')
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=self.early_stopping, verbose=1,
                                       mode='auto')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=self.lr_patience, verbose=1, factor=self.lr_factor,
                                      min_lr=1e-6)
        hist = model.fit_generator(batch_generator(x, y, self.batch_size),
                                   steps_per_epoch=self.steps_per_epoch,
                                   epochs=self.epochs,
                                   validation_data=(x_val, y_val),
                                   callbacks=[early_stopping, reduce_lr, save_best],
                                   verbose=2)  # one line per epoch
        # pandas.DataFrame(hist.history).to_csv(f'{PATH}/nn_logs/{path}.csv')
        epochs = len(hist.history['loss'])
        final_xent = hist.history['val_loss'][-1]
        return epochs, final_xent

    def fit(self, hp_key, x, y, x_val, y_val):
        path = uuid.uuid4().hex
        if self.multiclass_method == 'xent':
            epochs, final_xent = self.fit_model(hp_key, x, y, x_val, y_val, path)
        elif self.multiclass_method == 'ovr':
            epochs, final_xent = [], []
            for i in range(self.n_classes):
                y_ = (y == i).astype(int)
                y_val_ = (y_val == i).astype(int)
                epochs_, final_xent_ = self.fit_model(hp_key, x, y_, x_val, y_val_, f'{path}_{i}')
                epochs.append(epochs_)
                final_xent.append(final_xent_)
            epochs = numpy.mean(epochs)
            final_xent = numpy.mean(final_xent)
        return path, epochs, final_xent

    def score(self, path, x, y, prefix='train'):
        if self.multiclass_method == 'xent':
            model = keras.models.load_model(f'{PATH}/nn_weights/{path}.w')
            y_pred = model.predict(x)
            y_pred = numpy.argmax(y_pred, 1)

        elif self.multiclass_method == 'ovr':
            predictions = numpy.zeros((len(y), self.n_classes), dtype=float)
            for i in range(self.n_classes):
                model = keras.models.load_model(f'{PATH}/nn_weights/{path}_{i}.w')
                y_pred_ = model.predict(x)
                assert y_pred_.shape == (len(y), 2)
                predictions[:, i] = y_pred_[:, 1]
            y_pred = numpy.argmax(predictions, 1)

        return {f'{prefix}_score': accuracy_score(y, y_pred),
                f'{prefix}_f1': f1_score(y, y_pred, average='weighted')}

    def delete_weights(self, path):
        if self.multiclass_method == 'xent':
            os.remove(f'{PATH}/nn_weights/{path}.w')
        elif self.multiclass_method == 'ovr':
            for i in range(self.n_classes):
                os.remove(f'{PATH}/nn_weights/{path}_{i}.w')

    def __repr__(self):
        raise NotImplementedError


class Small_FCN(Base_Model):
    def get_model(self, **args):
        target_shape = (numpy.prod(self.shape),)
        l2 = keras.regularizers.l2(args['l2'])

        model = Sequential([
            Reshape(target_shape=target_shape, input_shape=self.shape),

            Dense(800, activation='relu', kernel_regularizer=l2, input_shape=(784,)),
            Dropout(.5),
            Dense(800, activation='relu', kernel_regularizer=l2),
            Dropout(.5),
            Dense(self.n_outputs, activation='softmax', kernel_regularizer=l2)
        ])

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        model.summary()

        return model

    def __repr__(self):
        if self.name:
            return self.name
        else:
            return 'neural-small-fcn'


class Regression_FCN(Base_Model):
    def get_model(self, **args):
        target_shape = (numpy.prod(self.shape),)
        l2 = keras.regularizers.l2(args['l2'])

        model = Sequential([
            Reshape(target_shape=target_shape, input_shape=self.shape),

            Dense(800, activation='relu', kernel_regularizer=l2, input_shape=(784,)),
            Dropout(.5),
            Dense(800, activation='relu', kernel_regularizer=l2),
            Dropout(.5),
            Dense(1, activation='linear', kernel_regularizer=l2)
        ])

        model.compile(loss='mse',
                      optimizer='adam', metrics=['accuracy'])
        model.summary()

        return model

    def __repr__(self):
        if self.name:
            return self.name
        else:
            return 'neural-regression-fcn'

    def score(self, path, x, y, prefix='train'):
        # model = keras.models.load_model(weight_path)
        # _, test_acc = model.evaluate(x, y)
        # return test_acc
        model = keras.models.load_model(f'{PATH}/nn_weights/{path}.w')

        y_ = model.predict(x)
        return {f'{prefix}_score': r2_score(y, y_),
                f'{prefix}_mae': mean_absolute_error(y, y_),
                f'{prefix}_mse': mean_squared_error(y, y_)}


class Neural_CNN(Base_Model):
    with_dense = True
    with_gap = False
    with_large = False

    def get_model(self, **args):
        input_shape = (*self.shape, 1)

        l2 = keras.regularizers.l2(args['l2'])

        layers = [
            Reshape(target_shape=input_shape, input_shape=self.shape),
            Conv2D(16, kernel_size=3, activation='relu', input_shape=input_shape, kernel_regularizer=l2),

            MaxPooling2D(pool_size=2),
            Conv2D(32, kernel_size=3, activation='relu', kernel_regularizer=l2),
        ]

        if self.with_large is True:
            layers += [
                MaxPooling2D(pool_size=2),
                Conv2D(64, kernel_size=3, activation='relu', kernel_regularizer=l2),

                MaxPooling2D(pool_size=2),
                Conv2D(128, kernel_size=3, activation='relu', kernel_regularizer=l2),
            ]

        if self.with_gap is True:
            layers += [
                GlobalAveragePooling2D(),
                Dropout(0.5),
            ]
        else:
            layers += [
                MaxPooling2D(pool_size=2),
                Flatten(),
                Dropout(0.5),
            ]

        if self.with_dense is True:
            layers += [
                Dense(128, activation='relu', kernel_regularizer=l2),
                Dropout(0.5),
            ]

        layers += [
            Dense(self.n_outputs, activation='softmax', kernel_regularizer=l2)
        ]

        model = Sequential(layers)

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        model.summary()

        return model

    def __repr__(self):
        if self.name:
            return self.name
        else:
            raise ValueError


class Neural_CNN_A(Neural_CNN):
    with_dense = True
    with_gap = False
    with_large = False


class Neural_CNN_B(Neural_CNN):
    with_dense = True
    with_gap = True
    with_large = False


class Neural_CNN_C(Neural_CNN):
    with_dense = False
    with_gap = False
    with_large = False


class Neural_CNN_D(Neural_CNN):
    with_dense = False
    with_gap = True
    with_large = False


class Neural_CNN_AL(Neural_CNN):
    with_dense = True
    with_gap = False
    with_large = True


class Neural_CNN_BL(Neural_CNN):
    with_dense = True
    with_gap = True
    with_large = True


class Neural_CNN_CL(Neural_CNN):
    with_dense = False
    with_gap = False
    with_large = True


class Neural_CNN_DL(Neural_CNN):
    with_dense = False
    with_gap = True
    with_large = True
