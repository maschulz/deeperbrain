import logging
import pickle


class FromFile_Data:
    def __init__(self, path):
        self.name = path.split('/')[-1].split('.')[-2].replace('_', '-')
        self.path = path

    def __call__(self):
        logging.debug(f'start loading file {self.path}')
        x, y, idx = pickle.load(open(self.path, 'rb'))
        logging.debug(f'done loading file {self.path}')
        return x, y, idx

    def __repr__(self):
        return self.name


DATA = {'mnist': FromFile_Data('data/mnist.p'),
        'fashion': FromFile_Data('data/fashion.p'),
        'superconductivity': FromFile_Data('data/superconductivity.p'),
        'tissue': FromFile_Data('data/tissue.p'),
        }
