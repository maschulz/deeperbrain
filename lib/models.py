import logging

import numpy
from hyperopt import fmin, tpe, hp, STATUS_OK
from numpy.linalg.linalg import LinAlgError
from sklearn.dummy import DummyClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, accuracy_score
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC

from lib.grid import SCALE
from lib.neural import Small_FCN, Regression_FCN, Neural_CNN_A, Neural_CNN_B, Neural_CNN_C, Neural_CNN_D, Neural_CNN_AL, \
    Neural_CNN_BL, Neural_CNN_CL, Neural_CNN_DL


class Base_Model:
    def run_grid(self, x, y, train_idx, val_idx, test_idx, grid, hyperparameters):
        for hp_key in ParameterGrid(grid):
            val_score = hyperparameters.get(hp_key)

            if val_score is None:
                model = self.get_model(**hp_key)

                self.fit(model, x[train_idx], y[train_idx], x[val_idx], y[val_idx])
                train_score = self.score(model, x[train_idx], y[train_idx], 'train')
                val_score = self.score(model, x[val_idx], y[val_idx], 'val')

                hp_result = {**hp_key, **train_score, **val_score}
                hyperparameters.set(hp_result)
                logging.debug(f'hp_trial {hp_result}')

        # FIXME: make sure to ONLY use HPs in specified grid
        hp_key = hyperparameters.best()
        hp_args = {hp: hp_key[hp] for hp in grid}

        model = self.get_model(**hp_args)
        self.fit(model, x[train_idx], y[train_idx], x[val_idx], y[val_idx])

        test_score = self.score(model, x[test_idx], y[test_idx], 'test')
        hp_key.update(test_score)

        return hp_key

    def run_hyperopt(self, x, y, train_idx, val_idx, test_idx, grid, trials):
        print(len(train_idx))

        def objective(hp_key):
            model = self.get_model(**hp_key)

            self.fit(model, x[train_idx], y[train_idx], x[val_idx], y[val_idx])
            train_score = self.score(model, x[train_idx], y[train_idx], 'train')
            val_score = self.score(model, x[val_idx], y[val_idx], 'val')

            hp_result = {**hp_key, **train_score, **val_score}
            hp_result['loss'] = -hp_result['val_score']
            hp_result['status'] = STATUS_OK
            logging.debug(f'hp_trial {hp_result}')
            return hp_result

        space = {}
        for p in grid:
            if SCALE[p] == 'lin' and len(grid[p]) > 1:
                space[p] = hp.uniform(p, grid[p][0], grid[p][-1])
            elif SCALE[p] == 'log' and len(grid[p]) > 1:
                space[p] = hp.loguniform(p, numpy.log(grid[p][0]), numpy.log(grid[p][-1]))
            elif len(grid[p]) == 1:
                space[p] = grid[p][0]
            else:
                raise ValueError

        best = fmin(objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=500,
                    trials=trials)

        model = self.get_model(**best)
        self.fit(model, x[train_idx], y[train_idx], x[val_idx], y[val_idx])

        test_score = self.score(model, x[test_idx], y[test_idx], 'test')
        best.update(test_score)

        return best

    def get_model(self, **args):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def fit(self, model, x_train, y_train, x_val, y_val):
        model.fit(x_train, y_train)

    def score(self, model, x, y, prefix='train'):
        y_ = model.predict(x)
        return {f'{prefix}_score': accuracy_score(y, y_),
                f'{prefix}_f1': f1_score(y, y_, average='weighted')}


class Regression_Mixin:
    def score(self, model, x, y, prefix='train'):
        y_ = model.predict(x)
        return {f'{prefix}_score': r2_score(y, y_),
                f'{prefix}_mae': mean_absolute_error(y, y_),
                f'{prefix}_mse': mean_squared_error(y, y_)}


########################################################################################################################
# linear models

class MajorityClassifier_Model(Base_Model):
    def get_model(self, **args):
        return DummyClassifier(strategy='most_frequent')

    def __repr__(self):
        return f'majority'


class LogisticRegression_Model(Base_Model):
    def get_model(self, **args):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(solver='lbfgs', max_iter=100, **args)

    def __repr__(self):
        return f'logisticregression'


class LDA_Model(Base_Model):
    def get_model(self, **args):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        # regularized lda with automatic shrinkage using the Ledoit-Wolf lemma
        # for cases where p > n
        return LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', **args)

    def __repr__(self):
        return f'lda'


class PrimalSVM_Model(Base_Model):
    def get_model(self, **args):
        from sklearn.svm import LinearSVC
        # limited to the default value of 1000 iterations (fails to converge in some cases)
        return LinearSVC(dual=False, **args)

    def __repr__(self):
        return f'svm-primal'


########################################################################################################################
# decision tree models

class RandomForest_Model(Base_Model):
    def get_model(self, **args):
        args['n_estimators'] = int(args['n_estimators'])
        args['max_depth'] = int(args['max_depth'])
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**args)

    def __repr__(self):
        return f'rforest'


class ExtraTrees_Model(Base_Model):
    def get_model(self, **args):
        args['n_estimators'] = int(args['n_estimators'])
        args['max_depth'] = int(args['max_depth'])
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(**args)

    def __repr__(self):
        return f'etrees'


class GradientBoostingEarlyStopping_Model(Base_Model):
    def get_model(self, **args):
        args['n_estimators'] = int(args['n_estimators'])
        args['max_depth'] = int(args['max_depth'])
        import xgboost
        return xgboost.XGBClassifier(**args, )

    def fit(self, model, x_train, y_train, x_val, y_val):
        if len(numpy.unique(y_train)) == 2:
            metric = 'logloss'
        else:
            metric = 'mlogloss'
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric=metric,
                  early_stopping_rounds=10, verbose=True)

    def __repr__(self):
        return f'xgb-es'


########################################################################################################################
# kernel models

from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel, sigmoid_kernel


def calculate_gram_matrix(x, kernel='linear', gamma=0, degree=0, coef0=0):
    if kernel == 'linear':
        gram = linear_kernel(x, x)
    elif kernel == 'poly':
        gram = polynomial_kernel(x, x, degree=degree, gamma=gamma, coef0=coef0)
    elif kernel == 'sigmoid':
        gram = sigmoid_kernel(x, x, gamma=gamma, coef0=coef0)
    elif kernel == 'rbf':
        gram = rbf_kernel(x, x, gamma=gamma)
    else:
        raise ValueError

    return gram


class KernelSVM_Model:
    name_of_complexity_HP = 'C'

    def __init__(self, kernel='linear'):
        self.kernel = kernel

    def run_grid(self, x, y, train_idx, val_idx, test_idx, grid, hyperparameters):
        grid = grid.copy()
        c_grid = grid.pop(self.name_of_complexity_HP)

        for hp_key in ParameterGrid(grid):
            gram = calculate_gram_matrix(x, kernel=self.kernel, **hp_key)

            for hp_c in c_grid:
                hp_key[self.name_of_complexity_HP] = hp_c
                val_score = hyperparameters.get(hp_key)

                if val_score is None:
                    model = self.get_model(**hp_key)

                    try:
                        gram_ = gram[numpy.ix_(train_idx, train_idx)]
                        model.fit(gram_, y[train_idx])
                        gram_ = gram[numpy.ix_(train_idx, train_idx)]
                        train_score = self.score(model, gram_, y[train_idx], 'train')
                        gram_ = gram[numpy.ix_(val_idx, train_idx)]
                        val_score = self.score(model, gram_, y[val_idx], 'val')

                        hp_result = {**hp_key, **train_score, **val_score}
                        hyperparameters.set(hp_result)
                        logging.debug(f'hp_trial {hp_result}')
                    except (ValueError, LinAlgError) as err:
                        logging.error(f'error in {hp_key}')
                        print(err)

        hp_key = hyperparameters.best()
        hp_args = {hp: hp_key[hp] for hp in list(grid.keys())}
        gram = calculate_gram_matrix(x, kernel=self.kernel, **hp_args)
        hp_args[self.name_of_complexity_HP] = hp_key[self.name_of_complexity_HP]
        model = self.get_model(**hp_args)

        gram_ = gram[numpy.ix_(train_idx, train_idx)]
        model.fit(gram_, y[train_idx])
        gram_ = gram[numpy.ix_(test_idx, train_idx)]
        test_score = self.score(model, gram_, y[test_idx], 'test')
        hp_key.update(test_score)

        return hp_key

    def run_hyperopt(self, x, y, train_idx, val_idx, test_idx, grid, trials):
        def objective(hp_key):
            model = self.get_model_hyperopt(**hp_key)

            model.fit(x[train_idx], y[train_idx])
            train_score = self.score(model, x[train_idx], y[train_idx], 'train')
            val_score = self.score(model, x[val_idx], y[val_idx], 'val')

            hp_result = {**hp_key, **train_score, **val_score}
            hp_result['loss'] = -hp_result['val_score']
            hp_result['status'] = STATUS_OK
            logging.debug(f'hp_trial {hp_result}')
            return hp_result

        space = {}
        for p in grid:
            if SCALE[p] == 'lin' and len(grid[p]) > 1:
                space[p] = hp.uniform(p, grid[p][0], grid[p][-1])
            elif SCALE[p] == 'log' and len(grid[p]) > 1:
                space[p] = hp.loguniform(p, numpy.log(grid[p][0]), numpy.log(grid[p][-1]))
            elif len(grid[p]) == 1:
                space[p] = grid[p][0]
            else:
                raise ValueError

        best = fmin(objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=500,
                    trials=trials)

        model = self.get_model_hyperopt(**best)
        model.fit(x[train_idx], y[train_idx])

        test_score = self.score(model, x[test_idx], y[test_idx], 'test')
        best.update(test_score)

        return best

    def get_model(self, **args):
        return SVC(kernel='precomputed', max_iter=1000, **args)

    def get_model_hyperopt(self, **args):
        return SVC(kernel=self.kernel, max_iter=1000, **args)

    def __repr__(self):
        return f'svm-{self.kernel}'

    def score(self, model, x, y, prefix='train'):
        y_ = model.predict(x)
        return {f'{prefix}_score': accuracy_score(y, y_),
                f'{prefix}_f1': f1_score(y, y_, average='weighted')}


########################################################################################################################
# regression models

class KernelRidge_Model(Regression_Mixin, KernelSVM_Model):
    name_of_complexity_HP = 'alpha'

    def get_model(self, **args):
        from sklearn.kernel_ridge import KernelRidge
        return KernelRidge(kernel='precomputed', **args)

    def get_model_hyperopt(self, **args):
        from sklearn.kernel_ridge import KernelRidge
        return KernelRidge(kernel=self.kernel, **args)

    def __repr__(self):
        return f'kernelridge-{self.kernel}'


class Ridge_Model(Regression_Mixin, Base_Model):
    def get_model(self, **args):
        from sklearn.linear_model import Ridge
        return Ridge(**args)

    def __repr__(self):
        return f'ridge'


class Lasso_Model(Regression_Mixin, Base_Model):
    def get_model(self, **args):
        from sklearn.linear_model import Lasso
        return Lasso(**args)

    def __repr__(self):
        return f'lasso'


class OLS_Model(Regression_Mixin, Base_Model):
    def get_model(self, **args):
        from sklearn.linear_model import LinearRegression
        return LinearRegression(**args)

    def __repr__(self):
        return f'ols'


########################################################################################################################

MODELS = {'majority': MajorityClassifier_Model(),
          'logisticregression': LogisticRegression_Model(),
          'lda': LDA_Model(),
          'svm-primal': PrimalSVM_Model(),

          'svm-rbf': KernelSVM_Model(kernel='rbf'),
          'svm-poly': KernelSVM_Model(kernel='poly'),
          'svm-sigmoid': KernelSVM_Model(kernel='sigmoid'),

          'ols': OLS_Model(),
          'ridge': Ridge_Model(),
          'lasso': Lasso_Model(),

          'kernelridge-rbf': KernelRidge_Model(kernel='rbf'),
          'kernelridge-poly': KernelRidge_Model(kernel='poly'),
          'kernelridge-sigmoid': KernelRidge_Model(kernel='sigmoid'),

          'rforest': RandomForest_Model(),
          'etrees': ExtraTrees_Model(),
          'xgb': GradientBoostingEarlyStopping_Model(),

          'neural-small-fcn': Small_FCN(),
          'neural-regression-fcn': Regression_FCN(),
          'neural-small-fcn-ovr': Small_FCN(multiclass_method='ovr', name="neural-small-fcn-ovr"),

          'neural-cnn-a': Neural_CNN_A(name="neural-cnn-a"),
          'neural-cnn-b': Neural_CNN_B(name="neural-cnn-b"),
          'neural-cnn-c': Neural_CNN_C(name="neural-cnn-c"),
          'neural-cnn-d': Neural_CNN_D(name="neural-cnn-d"),
          'neural-cnn-al': Neural_CNN_AL(name="neural-cnn-al"),
          'neural-cnn-bl': Neural_CNN_BL(name="neural-cnn-bl"),
          'neural-cnn-cl': Neural_CNN_CL(name="neural-cnn-cl"),
          'neural-cnn-dl': Neural_CNN_DL(name="neural-cnn-dl"),
          'neural-cnn-a-ovr': Neural_CNN_A(multiclass_method='ovr', name="neural-cnn-a-ovr"),
          'neural-cnn-b-ovr': Neural_CNN_B(multiclass_method='ovr', name="neural-cnn-b-ovr"),
          'neural-cnn-c-ovr': Neural_CNN_C(multiclass_method='ovr', name="neural-cnn-c-ovr"),
          'neural-cnn-d-ovr': Neural_CNN_D(multiclass_method='ovr', name="neural-cnn-d-ovr"),
          'neural-cnn-al-ovr': Neural_CNN_AL(multiclass_method='ovr', name="neural-cnn-al-ovr"),
          'neural-cnn-bl-ovr': Neural_CNN_BL(multiclass_method='ovr', name="neural-cnn-bl-ovr"),
          'neural-cnn-cl-ovr': Neural_CNN_CL(multiclass_method='ovr', name="neural-cnn-cl-ovr"),
          'neural-cnn-dl-ovr': Neural_CNN_DL(multiclass_method='ovr', name="neural-cnn-dl-ovr"),
          }
