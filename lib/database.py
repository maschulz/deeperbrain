import os
import pickle

import pandas
from hyperopt import Trials


class Database:
    def __init__(self, experiment_name: str, table_name: str):
        raise NotImplementedError

    def get(self, key: dict) -> dict:
        raise NotImplementedError

    def set(self, key: dict) -> None:
        raise NotImplementedError

    def best(self, key: dict, metric='score') -> dict:
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def __call__(self, filter_dict):
        return View(database=self, filter_dict=filter_dict)


# this was originally a mongodb interface...
class PandasDatabase(Database):
    def __init__(self, experiment_name, table_name, base_path='results'):
        self.file_name = base_path + '/' + experiment_name + '.' + table_name
        if os.path.exists(self.file_name):
            self.data = pandas.read_csv(self.file_name)
        else:
            self.data = pandas.DataFrame(columns=['sample_size', 'seed'])

        # load legacy data
        _, legacy_seed, legacy_table = self.file_name.split('.')
        legacy_file = base_path + '_old/' + experiment_name + '.' + legacy_table
        if os.path.exists(legacy_file):
            print('loading', legacy_file)
            legacy_data = pandas.read_csv(legacy_file)
            legacy_data = legacy_data[legacy_data.seed == int(legacy_seed)]
            self.data = self.data.append(legacy_data, ignore_index=True)
            self.data = self.data.drop_duplicates()

    def get(self, key):
        if len(self.data) == 0: return None
        data = self.data.loc[(self.data[list(key)] == pandas.Series(key)).all(axis=1)]
        if len(data) > 0:
            return data
        else:
            return None

    def set(self, key):
        self.data = self.data.append(key, ignore_index=True)

    def save(self):
        self.data.to_csv(self.file_name, index=False)

    def best(self, key, metric='val_score'):
        data = self.get(key)
        data = data.loc[data[metric].idxmax()]
        return dict(data)


class View:
    def __init__(self, database: Database, filter_dict: dict):
        self.filter = filter_dict
        self.database = database

    def get(self, key):
        key = key.copy()
        key.update(self.filter)
        return self.database.get(key)

    def set(self, key):
        key = key.copy()
        key.update(self.filter)
        self.database.set(key)

    def best(self, metric='val_score'):
        return self.database.best(self.filter, metric)


class TrialsFile:
    def __init__(self, experiment_name, table_name, base_path='results'):
        self.file_name = base_path + '/' + experiment_name + '.' + table_name
        if os.path.exists(self.file_name):
            self.trials = pickle.load(open(self.file_name, 'rb'))
        else:
            self.trials = {}

    def get(self, sample_size, seed):
        k = (sample_size, seed)
        if not k in self.trials:
            self.trials[k] = Trials()
        return self.trials[k]

    def save(self):
        pickle.dump(self.trials, open(self.file_name, 'wb'))
