import os
import pickle
from glob import glob

import numpy
from nilearn.datasets import load_mni152_template
from nilearn.image import resample_to_img
from nilearn.image import smooth_img

PATH = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = '/hpcwork/brainml/UKBB/sMRI/*.nii.gz'
N_JOBS = 20  # >20 seems to be problematic on rwth-hpc dialog nodes

F = glob(DATA_PATH)
N = [int(i.split('/')[-1].split('_')[0]) for i in F]
print('loaded ukbb files')

template = load_mni152_template()


def central_slices(a):
    dx, dy, dz = numpy.asarray(a.shape) // 2
    x = a.get_data()[dx, :, :]
    y = a.get_data()[:, dy, :]
    z = a.get_data()[:, :, dz]
    return (x, y, z)


def process(fn):
    print('processing:', fn)
    img = resample_to_img(fn, template)
    img = smooth_img(img, 8)  # disabled for some subanalyses
    slices = central_slices(img)
    return slices


from joblib import Parallel, delayed

X = Parallel(n_jobs=20)(delayed(process)(fn) for fn in F)

pickle.dump({'data': X, 'name': N}, open(PATH + '/ukbb_slices.p', 'wb'))
print('saved dataset to drive.')
