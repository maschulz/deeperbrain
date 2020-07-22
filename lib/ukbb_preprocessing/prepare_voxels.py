import sys
from glob import glob

import h5py
import numpy
from nilearn.datasets import load_mni152_template
from nilearn.image import resample_to_img
from nilearn.image import smooth_img
from nilearn.input_data import NiftiMasker

DATA_PATH = '/hpcwork/brainml/UKBB/sMRI/*.nii.gz'
MASK_PATH = '/hpcwork/brainml/SCZ/grey10_icbm_3mm_bin.nii.gz'
OUT_PATH = 'ukbb_masked.h5'


def prepare():
    F = glob(DATA_PATH)
    N = [int(i.split('/')[-1].split('_')[0]) for i in F]
    print('loaded ukbb files')

    template = load_mni152_template()
    mask_img = MASK_PATH
    m = NiftiMasker(mask_img=mask_img)
    m.fit()
    dim = numpy.sum(m.mask_img_.get_data())
    print('fitted grey matter mask')

    with h5py.File(OUT_PATH, "a") as store:
        if 'name' in store:
            dname = store['name']
            ddata = store['data']
        else:
            dname = store.create_dataset('name', (0,), maxshape=(None,), dtype='i')
            ddata = store.create_dataset('data', (0, dim), maxshape=(None, dim), dtype='f')
        print('opened hdf5 storage')

        done = dname[:]
        for fn, sn in zip(F, N):
            if sn in done:
                print(sn, 'exists')
                continue

            r = resample_to_img(fn, template)
            s = smooth_img(r, 8)  # disabled for some subanalyses
            x = m.transform(s)

            try:
                i = dname.shape[0]
                dname.resize(i + 1, axis=0)
                ddata.resize(i + 1, axis=0)

                ddata[i] = x
                dname[i] = sn
                print(sn, 'processed')

            except:
                dname.resize(i - 1, axis=0)
                ddata.resize(i - 1, axis=0)
                print('roll back changes and exit')
                sys.exit()

    print('finished')


prepare()
