import os
import pickle
from glob import glob

from nilearn.datasets import load_mni152_template
from nilearn.image import resample_to_img
from nilearn.image import smooth_img

PATH = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = '/hpcwork/brainml/UKBB/sMRI/*.nii.gz'
N_JOBS = 20  # >20 seems to be problematic on rwth-hpc dialog nodes

F = glob(DATA_PATH)
N = [int(i.split('/')[-1].split('_')[0]) for i in F]

template = load_mni152_template()


def process(fn):
    print('processing:', fn)
    img = resample_to_img(fn, template)
    img = smooth_img(img, 8)  # disabled for some subanalyses
    return img.get_data()


from joblib import Parallel, delayed

X = Parallel(n_jobs=20)(delayed(process)(fn) for fn in F)

# pickle.dump({'data': X, 'name': N}, open(PATH + '/ukbb_3d.p', 'wb'))
pickle.dump({'data': X, 'name': N}, open('/hpcwork/ms883464/ukbb_3d.p', 'wb'))
print('saved dataset to drive.')
