import logging

import numpy

from lib.config import Config
from lib.database import PandasDatabase as Database
from lib.database import TrialsFile


def run(config: Config):
    x, y, idx = config.data_func()
    x.flags.writeable = False  # for peace of mind...
    y.flags.writeable = False

    if config.sample_sizes is not None:
        sample_sizes = config.sample_sizes
    else:
        sample_sizes = sorted(idx.keys())

    for seed_ in config.seeds:

        if config.hyperopt is True:
            results = Database(experiment_name=config.parse(), table_name=f'{seed_}.hy_res')
            trials = TrialsFile(experiment_name=config.parse(), table_name=f'{seed_}.hy_hp')
        else:
            results = Database(experiment_name=config.parse(), table_name=f'{seed_}.res')
            hyperparameters = Database(experiment_name=config.parse(grid=False), table_name=f'{seed_}.hp')

        for sample_size_ in sample_sizes:
            assert sample_size_ in idx

            key = {'sample_size': sample_size_, 'seed': seed_}

            result = results.get(key)
            if result is None:
                train_idx, val_idx, test_idx = idx[sample_size_][seed_]
                all_idx = numpy.concatenate((train_idx, val_idx, test_idx))

                x_ = x[all_idx]
                y_ = y[all_idx]

                train_idx_ = numpy.arange(len(train_idx))
                val_idx_ = numpy.arange(len(train_idx), len(train_idx) + len(val_idx))
                test_idx_ = numpy.arange(len(train_idx) + len(val_idx), len(train_idx) + len(val_idx) + len(test_idx))

                x_ = config.scaling_func(x_, train_idx_)
                x_ = config.dim_reduction_func(x_, y_, train_idx_)

                if config.hyperopt is True:
                    result = config.classifier_func.run_hyperopt(x_, y_, train_idx_, val_idx_, test_idx_, config.grid(),
                                                                 trials.get(sample_size_, seed_))
                    trials.save()
                else:
                    result = config.classifier_func.run_grid(x_, y_, train_idx_, val_idx_, test_idx_, config.grid(),
                                                             hyperparameters(key))
                    hyperparameters.save()

                key.update(result)
                results.set(key)
                results.save()

                logging.info(f'{config.parse()}\t{key}')

            else:
                # only for keeping legacy results
                if config.hyperopt is not True:
                    results.save()
                    hyperparameters.save()

                logging.info(f'{config.parse()}\tn={sample_size_}\ts={seed_}\tskipped')
