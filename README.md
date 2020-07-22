<div align="center"> 
   
# Deep Learning for Brains
[![Preprint](http://img.shields.io/badge/preprint-bioRxiv%3A757054-B31B1B.svg)](https://www.biorxiv.org/content/10.1101/757054v1)

</div>

> Schulz, M.A., Yeo, T., Vogelstein, J., Mourao-Miranada, J., Kather, J., Kording, K., Richards, B.A. and Bzdok, D., 2019. Deep learning for brains?: Different linear and nonlinear scaling in UK Biobank brain images vs. machine-learning datasets. bioRxiv, p.757054.

1) create conda environment\
`conda env create --prefix .envs/deeperbrain_public -f env.yaml`

2) activate environment\
`source activate .envs/deeperbrain_public`

3) prepare datasets\
`python prepare_datasets.py mnist` \
This should work for the publicly available datasets MNIST, Fashion, Tissue (Kather et al. 2019), Superconductivity (Hamidieh et al. 2018). UK Biobank data ist not public, but you can find details on our preprocessing in `lib/ukbb_preprocessing`.

4) run analyses, e.g.:\
`python run.py --data mnist --model logisticregression --grid v3`

5) aggregate results to csv file\
`python aggregate_results.py`

6) plot, e.g.:\
`python plot.py mnist --grid v3`