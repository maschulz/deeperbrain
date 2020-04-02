import numpy


def logrange(start, stop, step=5., base=2.):
    base = float(base)
    return numpy.power(base, numpy.arange(start, stop + step, step))


class Grid:
    def __init__(self, model, version):
        self.name = version
        self.grid = GRIDS[version][model]

    def __call__(self):
        return self.grid

    def __repr__(self):
        return f'{self.name}'


SCALE = {'C': 'log',
         'gamma': 'log',
         'coef0': 'lin',
         'degree': 'lin',
         'alpha': 'log',
         'l2': 'log',
         'max_features': 'lin',
         'max_depth': 'lin',
         'subsample': 'lin',
         'learning_rate': 'log',
         'colsample_bytree': 'lin',
         }

grid_n = lambda n: {
    'lda': {},
    'gaussiannb': {},
    'bernoullinb': {},
    'majority': {},

    'logisticregression': {'C': logrange(-20, 10, n)},
    'logisticregression-m': {'C': logrange(-20, 10, n)},
    'svm-primal': {'C': logrange(-20, 10, n)},

    'svm-rbf': {'C': logrange(-10, 20, n), 'gamma': logrange(-25, 5, n)},
    'svm-sigmoid': {'C': logrange(-10, 20, n), 'gamma': logrange(-25, 5, n), 'coef0': [-1, 0, 1, ]},
    'svm-poly': {'C': logrange(-10, 20, n), 'gamma': logrange(-25, 5, n), 'coef0': [-1, 0, 1], 'degree': [2, ]},

    'ols': {},
    'ridge': {'alpha': logrange(-15, 15, n)},
    'lasso': {'alpha': logrange(-15, 15, n)},

    'kernelridge-rbf': {'alpha': logrange(-15, 15, n), 'gamma': logrange(-25, 5, n)},
    'kernelridge-sigmoid': {'alpha': logrange(-15, 15, n), 'gamma': logrange(-25, 5, n), 'coef0': [-1, 0, 1, ]},
    'kernelridge-poly': {'alpha': logrange(-15, 15, n), 'gamma': logrange(-25, 5, n), 'coef0': [-1, 0, 1],
                         'degree': [2, ]},

    'neural-small-fcn': {'l2': logrange(-4, -0.5, 0.5 * n, 10)},
    'neural-small-fcn-ovr': {'l2': logrange(-4, -0.5, 0.5 * n, 10)},
    'neural-regression-fcn': {'l2': logrange(-4, -0.5, 0.5 * n, 10)},

    'neural-cnn-a': {'l2': [0.]},
    'neural-cnn-b': {'l2': [0.]},
    'neural-cnn-c': {'l2': [0.]},
    'neural-cnn-d': {'l2': [0.]},
    'neural-cnn-al': {'l2': [0.]},
    'neural-cnn-bl': {'l2': [0.]},
    'neural-cnn-cl': {'l2': [0.]},
    'neural-cnn-dl': {'l2': [0.]},
    'neural-cnn-a-ovr': {'l2': [0.]},
    'neural-cnn-b-ovr': {'l2': [0.]},
    'neural-cnn-c-ovr': {'l2': [0.]},
    'neural-cnn-d-ovr': {'l2': [0.]},
    'neural-cnn-al-ovr': {'l2': [0.]},
    'neural-cnn-bl-ovr': {'l2': [0.]},
    'neural-cnn-cl-ovr': {'l2': [0.]},
    'neural-cnn-dl-ovr': {'l2': [0.]},

    'rforest': {'n_estimators': [1000, ],
                'max_features': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0],
                'max_depth': [4, 6, 8, 10, 10000],
                },
    'etrees': {'n_estimators': [1000, ],
               'max_features': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0],
               'max_depth': [4, 6, 8, 10, 10000],
               },
    'xgb': {'n_estimators': [2000, ],
            'subsample': [0.5, 1.0, ],
            'learning_rate': [0.1, ],
            'colsample_bytree': [1.0, 0.5],
            'max_depth': [2, 3, 4, 6, 8, 10, ],
            },

}

neural_a = {
    'neural-small-fcn': {'l2': logrange(-4, -0.5, 0.25, 10)},
    'neural-small-fcn-ovr': {'l2': logrange(-4, -0.5, 0.25, 10)},
    'neural-regression-fcn': {'l2': logrange(-4, -0.5, 0.25, 10)},

    'neural-cnn-a': {'l2': [0., 1e-3, 1e-5, 1e-7]},
    'neural-cnn-b': {'l2': [0., 1e-3, 1e-5, 1e-7]},
    'neural-cnn-c': {'l2': [0., 1e-3, 1e-5, 1e-7]},
    'neural-cnn-d': {'l2': [0., 1e-3, 1e-5, 1e-7]},
    'neural-cnn-al': {'l2': [0., 1e-3, 1e-5, 1e-7]},
    'neural-cnn-bl': {'l2': [0., 1e-3, 1e-5, 1e-7]},
    'neural-cnn-cl': {'l2': [0., 1e-3, 1e-5, 1e-7]},
    'neural-cnn-dl': {'l2': [0., 1e-3, 1e-5, 1e-7]},
    'neural-cnn-a-ovr': {'l2': [0., 1e-3, 1e-5, 1e-7]},
    'neural-cnn-b-ovr': {'l2': [0., 1e-3, 1e-5, 1e-7]},
    'neural-cnn-c-ovr': {'l2': [0., 1e-3, 1e-5, 1e-7]},
    'neural-cnn-d-ovr': {'l2': [0., 1e-3, 1e-5, 1e-7]},
    'neural-cnn-al-ovr': {'l2': [0., 1e-3, 1e-5, 1e-7]},
    'neural-cnn-bl-ovr': {'l2': [0., 1e-3, 1e-5, 1e-7]},
    'neural-cnn-cl-ovr': {'l2': [0., 1e-3, 1e-5, 1e-7]},
    'neural-cnn-dl-ovr': {'l2': [0., 1e-3, 1e-5, 1e-7]},
}

GRIDS = {
    'v1': grid_n(4),
    'v2': grid_n(2),
    'v3': grid_n(1),
    'v3-a': dict(grid_n(1), **neural_a),
}
