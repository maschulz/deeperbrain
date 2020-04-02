import argparse
import os

import matplotlib
import matplotlib.patches as mpatches
import pandas
import seaborn
from matplotlib import pylab

if not os.path.exists('plots'):
    os.makedirs('plots')

pylab.rc('font', family='serif', serif='Times')
# pylab.rc('text', usetex=True)
pylab.rc('xtick', labelsize=8)
pylab.rc('ytick', labelsize=8)
pylab.rc('axes', labelsize=8)

CI = 'sd'
NA = 'neural-cnn-a-ovr'
NB = 'neural-cnn-b-ovr'
NAL = 'neural-cnn-al-ovr'
NBL = 'neural-cnn-bl-ovr'
NF = 'neural-small-fcn-ovr'

color_dict = {NAL: 'blue',
              NBL: 'baby blue',
              NA: 'blue',
              NB: 'baby blue',
              NF: 'cyan',
              'neural-regression-fcn': 'cyan',
              'svm-rbf': 'olive',
              'svm-poly': 'yellow green',
              'svm-sigmoid': 'lawn green',
              'svm-primal': 'magenta',
              'logisticregression': 'purple',
              'lda': 'pink',
              'kernelridge-rbf': 'olive',
              'kernelridge-poly': 'yellow green',
              'kernelridge-sigmoid': 'lawn green',
              'ridge': 'purple',
              'lasso': 'magenta',
              'elastic': 'pink',
              'blank': 'white',
              'rforest': 'yellow',
              'etrees': 'mustard',
              'xgb': 'orange'
              }

name_dict = {NAL: 'Convolutional Neural Net',
             NBL: 'Convolutional Neural Net (GAP)',
             NA: 'Convolutional Neural Net',
             NB: 'Convolutional Neural Net (GAP)',
             NF: 'Fully Connected Neural Net',
             'neural-regression-fcn': 'Fully Connected Neural Net',
             'svm-rbf': 'Kernel SVM (rbf)',
             'svm-poly': 'Kernel SVM (polynomial)',
             'svm-sigmoid': 'Kernel SVM (sigmoidal)',
             'svm-primal': 'SVM (linear)',
             'logisticregression': 'Logistic Regression',
             'lda': 'Linear Discriminant Analysis',
             'kernelridge-rbf': 'Kernel Ridge (rbf)',
             'kernelridge-poly': 'Kernel Ridge (poly)',
             'kernelridge-sigmoid': 'Kernel Ridge (sigmoid)',
             'ridge': 'Ridge',
             'lasso': 'Lasso',
             'elastic': 'ElasticNet',
             'blank': '',
             'rforest': 'Random Forest',
             'etrees': 'Extremely Randomized Trees',
             'xgb': 'Gradient Boosted Trees'

             }


def plot(dataset, grid, trafo):
    assert os.path.exists('results.csv'), 'need to aggregate results first'

    df = pandas.read_csv('results.csv')
    df = df[(df.grid == grid) & (df.data == dataset) & (df.trafo == trafo)]

    fig, ax = pylab.subplots(1, 1, dpi=200)
    # fig.subplots_adjust(left=.1, bottom=.225, right=.98, top=.94, hspace=0.35, wspace=0.25)

    models = [NA, NB, NF,
              'svm-rbf', 'svm-poly', 'svm-sigmoid', 'svm-primal',
              'logisticregression', 'lda', 'rforest', 'etrees', 'xgb'
              ]

    names = [name_dict[i] for i in models]
    palette = [color_dict[i] for i in models]
    palette = seaborn.xkcd_palette(palette)
    legend = [mpatches.Patch(color=i, label=j) for i, j in zip(palette, names)]

    seaborn.lineplot(x='sample_size', y='test_score', hue='model', hue_order=models, data=df, ax=ax, palette=palette,
                     legend=False, err_style="bars", ci=CI)
    ax.set_title(f'Dataset: {dataset}', fontsize=8)

    ax.set_xscale('log')
    ax.set_xticks(df.sample_size.unique())
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_tick_params(which='minor', size=0)
    ax.get_xaxis().set_tick_params(which='minor', width=0)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Sample Size')
    # ax.set_ylim(0.65, 1)

    pylab.figlegend(handles=legend, ncol=2, fontsize=8, loc='lower center', frameon=False)

    fig.savefig(f'plots/{dataset}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dataset', metavar='dataset', type=str)
    parser.add_argument('--grid', default='v1', type=str)
    parser.add_argument('--trafo', default='identity', type=str)
    args = parser.parse_args()

    plot(args.dataset, args.grid, args.trafo)
