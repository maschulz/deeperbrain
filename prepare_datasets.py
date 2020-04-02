import argparse
import glob
import os
import pickle

import numpy as np
import pandas
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

if not os.path.exists('data'):
    os.makedirs('data')


def get_dataset(dataset):
    if dataset == 'mnist':
        from keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))
        # x = x.reshape(-1, 28 ** 2)
        x, _, y, _ = train_test_split(x, y, train_size=9300, stratify=y, random_state=42)
        return x, y

    elif dataset == 'mnist_binary':
        from keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))
        y = y % 2 == 0
        # x = x.reshape(-1, 28 ** 2)
        x, _, y, _ = train_test_split(x, y, train_size=9300, stratify=y, random_state=42)
        return x, y

    elif dataset == 'fashion':
        from keras.datasets import fashion_mnist
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))
        # x = x.reshape(-1, 28 ** 2)
        x, _, y, _ = train_test_split(x, y, train_size=9300, stratify=y, random_state=42)
        return x, y

    elif dataset == 'tissue':
        assert os.path.exists('data/NCT-CRC-HE-100K'), \
            """ first download
            https://zenodo.org/record/1214456/files/NCT-CRC-HE-100K.zip?download=1
            extract & move to data folder
            """

        F = glob.glob('data/NCT-CRC-HE-100K/*/*.tif')
        Y = [i.split('/')[1] for i in F]

        f, _, y, _ = train_test_split(F, Y, train_size=9300, stratify=Y, random_state=42)

        x = [np.asarray(Image.open(i).convert('L').resize((100, 100))) for i in tqdm(f)]
        x = np.stack(x)

        y_encoded = LabelEncoder().fit_transform(y)

        return x, y_encoded

    elif dataset == 'superconductivity':
        assert os.path.exists('data/superconductivity.csv'), \
            """ first download
            https://archive.ics.uci.edu/ml/machine-learning-databases/00464/superconduct.zip
            extract, move to data folder, and rename train.csv to superconductivity.csv for clarity
            """

        df = pandas.read_csv('superconductivity.csv')
        y = df.values[:, -1]
        x = df.values[:, :-1]
        x, _, y, _ = train_test_split(x, y, train_size=9300, random_state=42)
        return x, y

    else:
        raise ValueError('no such dataset')


def prep_dataset(dataset, do_stratify=True):
    x, y = get_dataset(dataset)

    idx = np.arange(min(len(y), 9300))
    IDX = {}

    for n in [100, 200, 500, 1000, 2000, 4000, 8000]:
        if len(y) - 1300 < n: continue
        IDX[n] = []
        for i in range(50):
            if do_stratify:
                stratify = y
            else:
                stratify = None
            idx_train, idx_test = train_test_split(idx, test_size=650, stratify=stratify, random_state=i)
            if do_stratify:
                stratify = y[idx_train]
            else:
                stratify = None
            idx_train, idx_val = train_test_split(idx_train, train_size=n, test_size=650, stratify=stratify,
                                                  random_state=i)
            IDX[n].append([idx_train, idx_val, idx_test])
    pickle.dump((x, y, IDX), open('data/%s.p' % dataset, 'wb'))
    print('prepared', dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dataset', metavar='dataset', type=str)
    args = parser.parse_args()

    prep_dataset(args.dataset)
