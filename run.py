import argparse
import logging
import os
import warnings

from sklearn.exceptions import DataConversionWarning, ConvergenceWarning

from lib.config import Config
from lib.data import DATA
from lib.grid import Grid, GRIDS
from lib.models import MODELS
from lib.preprocessing import TRAFOS, SCALING
from lib.trial import run

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DataConversionWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate dataset x model combination.')
    parser.add_argument('--data', default='mnist', type=str, choices=list(DATA.keys()))
    parser.add_argument('--model', default='majority', type=str, choices=list(MODELS.keys()))
    parser.add_argument('--grid', default='v1', type=str, choices=list(GRIDS.keys()))
    parser.add_argument('--trafo', default='identity', type=str, choices=list(TRAFOS.keys()))
    parser.add_argument('--scaling', default='standard', type=str, choices=list(SCALING.keys()))
    parser.add_argument('--dim', default=784, type=int)
    parser.add_argument('--seeds', nargs='+', default=range(10), type=int)
    parser.add_argument('--sample_sizes', nargs='+', type=int)
    parser.add_argument('--hyperopt', action="store_true", default=False)
    args = parser.parse_args()

    # TODO: set up folder structure if necessary
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('nn_weights'):
        os.makedirs('nn_weights')

    config = Config(DATA[args.data],
                    SCALING[args.scaling](),
                    TRAFOS[args.trafo](n_components=args.dim),
                    MODELS[args.model],
                    Grid(args.model, args.grid),
                    seeds=args.seeds,
                    sample_sizes=args.sample_sizes,
                    hyperopt=args.hyperopt
                    )

    run(config)
