class Config:
    def __init__(self, data_func, scaling_func, dim_reduction_func, classifier_func, grid, seeds,
                 sample_sizes, hyperopt):
        self.data_func = data_func
        self.scaling_func = scaling_func
        self.dim_reduction_func = dim_reduction_func
        self.classifier_func = classifier_func
        self.grid = grid
        self.sample_sizes = sample_sizes
        self.seeds = seeds
        self.hyperopt = hyperopt

    def parse(self, grid=True):
        s = f'{self.data_func}_{self.scaling_func}_{self.dim_reduction_func}_{self.classifier_func}'
        if grid:
            s += f'_{self.grid}'
        return s
