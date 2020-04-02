import argparse
import glob

import pandas
from joblib import Parallel, delayed
from tqdm import tqdm

from lib.grid import GRIDS


def process_file(f):
    try:
        df = pandas.read_csv(f)
    except:
        print(f'error in {f}')
        return None

    # extract info from filename
    d, s, t, m, g = f.split('/')[-1].split('.')[0].split('_')
    # add data as columns
    df['data'] = d
    df['scaling'] = s
    df['trafo'] = t
    df['model'] = m
    df['grid'] = g

    if d[:5] == 'slice':
        data_type, data_axis, _, _ = d.split('-')
        df['data_smoothing'] = d[:7] != 'sliceNS'
        df['data_type'] = 'slice'
        df['data_axis'] = data_axis
        df['target'] = d.split('-')[-1]

    if d[:5] == 'voxel':
        data_type, data_modality, _ = d.split('-')
        df['data_smoothing'] = d[:8] != 'voxelsNS'
        df['data_type'] = 'voxel'
        df['data_modality'] = data_modality
        df['target'] = d.split('-')[-1]

    return df


def read_results(grid='v3-a', hyperopt=False):
    if hyperopt is True:
        F = glob.glob('results/*.hy_res')
    else:
        F = glob.glob('results/*.res')

    dfs = Parallel(n_jobs=20)(delayed(process_file)(f) for f in tqdm(F))
    dfs = [i for i in dfs if i is not None]

    DF = pandas.concat(dfs, ignore_index=True)

    DF = DF[DF.model != 'majority']
    DF = DF[DF.model != 'gaussiannb']
    DF = DF[DF.model != 'bernoullinb']
    DF = DF[DF.grid == grid]

    return DF


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--grid', default='v1', type=str, choices=list(GRIDS.keys()))
    parser.add_argument('--hyperopt', action="store_true", default=False)
    args = parser.parse_args()

    DF = read_results(args.grid, hyperopt=args.hyperopt)
    DF = DF.drop_duplicates()
    DF.to_csv(f'''results{'.hyperopt' if args.hyperopt else ''}.csv''')
