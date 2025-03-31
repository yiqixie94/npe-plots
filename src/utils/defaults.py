import numpy as np

def get_corner_default(**upd):
    default = dict(
        bins=30, smooth=0.9, color='C0',
        truth_color='black', quantiles=[0.05, 0.95], levels=[0.5, 0.9],
        plot_density=False, plot_datapoints=False, fill_contours=True,
        max_n_ticks=3, 
    )
    default.update(upd)
    return default

def get_corner_default_bilby(**upd):
    default = dict(
        bins=50, smooth=0.9,
        title_kwargs=dict(fontsize=16), color='#0072C1',
        truth_color='tab:orange', quantiles=[0.16, 0.84],
        levels=(1-np.exp(-0.5), 1-np.exp(-2), 1-np.exp(-9/2.)),
        plot_density=False, plot_datapoints=True, fill_contours=True,
        max_n_ticks=3
    )
    default.update(upd)
    return default