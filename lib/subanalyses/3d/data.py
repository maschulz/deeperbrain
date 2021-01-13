import pickle

import h5py
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler


class HDF5Dataset(Dataset):
    def __init__(self, file_name, x_name, y_name, y_type=int):
        self.y_type = y_type
        data = h5py.File(file_name, 'r')
        self.x = data[x_name]
        self.y = data[y_name]

    def __getitem__(self, index):
        x_ = self.x[index]
        x = (x_.reshape((1,) + x_.shape) - MEAN) / STD
        y = self.y[index].astype(self.y_type)
        return x, y

    def __len__(self):
        return len(self.x)


def get_dataloaders(sample_size, seed, batch_size=4, target='ageXsex'):
    if target == 'ageXsex':
        regression = False
        output_dim = 10
        y_type = int
    elif target == 'sex':
        regression = False
        output_dim = 2
        y_type = int
    elif target == 'ageC':
        regression = True
        output_dim = 1
        y_type = float
    else:
        raise ValueError

    print('loading data')
    idx = pickle.load(open(IDX_PATH, 'rb'))
    idx_train, idx_val, idx_test = idx[sample_size][seed]
    data = HDF5Dataset(H5_PATH, 'mri', target, y_type)

    train_loader = DataLoader(data, batch_size=batch_size, sampler=SubsetRandomSampler(idx_train))
    val_loader = DataLoader(data, batch_size=batch_size, sampler=SubsetRandomSampler(idx_val))
    test_loader = DataLoader(data, batch_size=batch_size, sampler=SubsetRandomSampler(idx_test))
    print('loaded data')

    return {'train': train_loader, 'test': val_loader, 'val': test_loader}, regression, output_dim
