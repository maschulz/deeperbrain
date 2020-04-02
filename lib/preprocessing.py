import numpy


class Minmax_Scaler:
    def __call__(self, x, train_idx):
        x = x.reshape(len(x), -1)
        from sklearn.preprocessing import MinMaxScaler
        method = MinMaxScaler()
        method.fit(x[train_idx])
        return method.transform(x)

    def __repr__(self):
        return 'minmax'


class Standard_Scaler:
    def __call__(self, x, train_idx):
        x = x.reshape(len(x), -1)
        from sklearn.preprocessing import StandardScaler
        method = StandardScaler()
        method.fit(x[train_idx])
        return method.transform(x)

    def __repr__(self):
        return 'standard'


class Robust_Scaler:
    def __call__(self, x, train_idx):
        x = x.reshape(len(x), -1)
        from sklearn.preprocessing import RobustScaler
        method = RobustScaler()
        method.fit(x[train_idx])
        return method.transform(x)

    def __repr__(self):
        return 'robust'


class Image_Scaler:
    def __init__(self, train_idx):
        pass

    def __call__(self, x, train_idx):
        return (x - numpy.mean(x)) / numpy.std(x)

    def __repr__(self):
        return 'image'


class Identity_Transformer:
    def __init__(self, n_components=784):
        pass

    def __call__(self, x, y, train_idx):
        return x

    def __repr__(self):
        return 'identity'


class Noise_Transformer:
    def __init__(self, n_components=1):
        self.factor = n_components
        pass

    def __call__(self, x, y, train_idx):
        return x + self.factor * numpy.random.randn(*x.shape)

    def __repr__(self):
        return 'noise'


class PCA_Transformer:
    def __init__(self, n_components=100):
        self.n_components = n_components

    def __call__(self, x, y, train_idx):
        from sklearn.decomposition import PCA
        method = PCA(n_components=self.n_components, whiten=True)
        method.fit(x[train_idx])
        return method.transform(x)

    def __repr__(self):
        return f'pca-n{self.n_components}'


class RandomProjection_Transformer:
    def __init__(self, n_components=784):
        self.n_components = n_components

    def __call__(self, x, y, train_idx):
        from sklearn.random_projection import GaussianRandomProjection
        method = GaussianRandomProjection(n_components=self.n_components, random_state=42)
        method.fit(x[train_idx])
        x_t = method.transform(x)

        # need to rescale
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(x_t[train_idx])
        x_t = scaler.transform(x_t)

        return x_t

    def __repr__(self):
        return f'random-n{self.n_components}'


class RandomSubset_Transformer:
    def __init__(self, n_components=784):
        self.n_components = n_components

    def __call__(self, x, y, train_idx):
        n_features = x.shape[1]
        random_state = numpy.random.RandomState(42)
        idx_features = random_state.choice(n_features, size=self.n_components, replace=False)
        return x[:, idx_features]

    def __repr__(self):
        return f'subset-n{self.n_components}'


class UnivariateFeatureSelection_Transformer:
    def __init__(self, n_components=784):
        self.n_components = n_components

    def __call__(self, x, y, train_idx):
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import f_classif
        method = SelectKBest(f_classif, self.n_components)
        method.fit(x[train_idx], y[train_idx])
        return method.transform(x)

    def __repr__(self):
        return f'ufs-n{self.n_components}'


# class SparsePCA_Transformer:
#     def __init__(self, n_components=784, alpha=1):
#         self.alpha = alpha
#         self.n_components = n_components
#
#     def __call__(self, x, y, train_idx):
#         from sklearn.decomposition import SparsePCA
#         method = SparsePCA(n_components=self.n_components, alpha=self.alpha, normalize_components=True)
#         method.fit(x[train_idx])
#         return method.transform(x)
#
#     def __repr__(self):
#         return f'sparsepca-n{self.n_components}-a{self.alpha}'


class RecursiveFeatureElimination_Transformer:
    def __init__(self, n_components=784):
        self.n_components = n_components

    def __call__(self, x, y, train_idx):
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LogisticRegression
        # limited iterations & large step size to reduce runtime to max 15min on one core
        # logres on 8000 samples and 80.000 features runs ca 3 min
        # 3min x 4 RFE runs = 12min
        method = RFE(
            LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=15, verbose=2, random_state=42),
            n_features_to_select=self.n_components, step=0.25, )
        method.fit(x[train_idx], y[train_idx])
        return method.transform(x)

    def __repr__(self):
        return f'rfe-n{self.n_components}'


SCALING = {'standard': Standard_Scaler,
           'image': Image_Scaler,
           }

TRAFOS = {'pca': PCA_Transformer,
          'ufs': UnivariateFeatureSelection_Transformer,
          'random': RandomProjection_Transformer,
          'noise': Noise_Transformer,
          'rfe': RecursiveFeatureElimination_Transformer,
          'identity': Identity_Transformer,
          'subset': RandomSubset_Transformer,
          # 'sparsepca': SparsePCA_Transformer,
          }
