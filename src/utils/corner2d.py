import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import corner

from .defaults import get_corner_default
# from .defaults import get_corner_default_bilby


def plot_hist2d(
        x_data, y_data, 
        x_range=None, y_range=None,
        x_truth=None, y_truth=None, 
        bins=30, levels=[0.5,0.9], 
        ax=None, color='C0', truth_color='black', 
        truth_marker='s', truth_marker_size=30):
    if ax is None:
        ax = plt.gca()
    if x_range is None:
        x_range = [np.min(x_data), np.max(x_data)]
    elif type(x_range) == float:
        x_range = np.quantile(x_data, [(1-x_range)/2, (1+x_range)/2])
    if y_range is None:
        y_range = [np.min(y_data), np.max(y_data)]
    elif type(y_range) == float:
        y_range = np.quantile(y_data, [(1-y_range)/2, (1+y_range)/2])
    kwargs = get_corner_default(color=color, bins=bins, levels=levels)
    corner.hist2d(x_data, y_data, range=[x_range,y_range], ax=ax, **kwargs)
    if x_truth is not None and y_truth is not None:
        ax.scatter(x_truth, y_truth, c=truth_color, marker=truth_marker, s=truth_marker_size)
    if x_truth is not None:
        ax.axvline(x_truth, c=truth_color)
    if y_truth is not None:
        ax.axhline(y_truth, c=truth_color)
    return ax


def plot_hist(
        data, data_range=None, truth=None,
        bins=30, quantiles=[0.05,0.95],
        ax=None, color='C0', truth_color='black', orientation='vertical'):
    if ax is None:
        ax = plt.gca()
    if data_range is None:
        data_range = [np.min(data), np.max(data)]
    elif type(data_range) == float:
        data_range = np.quantile(data, [(1-data_range)/2, (1+data_range)/2])
    bins = np.linspace(*data_range, bins+1)
    ax.hist(data, bins=bins, density=True, 
            color=color, histtype='step', orientation=orientation)
    plot_line = ax.axvline if orientation == 'vertical' else ax.axhline
    l, r = np.quantile(data, quantiles)
    plot_line(l, c=color, linestyle='dashed')
    plot_line(r, c=color, linestyle='dashed')
    if truth is not None:
        plot_line(truth, c=truth_color)
    return ax


def initialize_figure_for_corner2d(fig, subplots_ratio=1., subplots_space=None):
    fig.subplots(2, 2, 
                 width_ratios=[subplots_ratio, 1.],
                 height_ratios=[1., subplots_ratio])
    fig.subplots_adjust(wspace=subplots_space, hspace=subplots_space)
    return fig


def finalize_figure_for_corner2d(fig, x_label=None, y_label=None):
    fig.axes[2].set_xlabel(x_label)
    fig.axes[2].set_ylabel(y_label)
    fig.axes[0].set_xticks([])
    fig.axes[0].set_yticks([])
    fig.axes[2].autoscale(True)
    fig.canvas.draw_idle()
    fig.axes[2].autoscale(False)
    fig.axes[0].set_xlim(fig.axes[2].get_xlim())
    fig.axes[3].set_xticks([])
    fig.axes[3].set_yticks([])
    fig.axes[3].set_ylim(fig.axes[2].get_ylim())
    fig.axes[1].axis('off')
    return fig


def plot_corner2d(
        x_data, y_data, 
        x_range=None, y_range=None, 
        x_truth=None, y_truth=None, 
        x_label=None, y_label=None,
        bins=30, levels=[0.5,0.9], quantiles=[0.05,0.95],
        fig=None, initialize_fig=True, finalize_fig=True,
        subplots_ratio=1., subplots_space=None, 
        color='C0', truth_color='black', 
        truth_marker='s', truth_marker_size=30):
    
    if fig is None:
        fig = plt.figure()
    if initialize_fig:
        initialize_figure_for_corner2d(fig, subplots_ratio, subplots_space)

    plot_hist2d(x_data, y_data, 
                x_range=x_range, y_range=y_range,
                x_truth=x_truth, y_truth=y_truth,
                bins=bins, levels=levels, 
                ax=fig.axes[2], color=color, truth_color=truth_color, 
                truth_marker=truth_marker, truth_marker_size=truth_marker_size)
    plot_hist(x_data, data_range=x_range, truth=x_truth, bins=bins, quantiles=quantiles,
              ax=fig.axes[0], color=color, truth_color=truth_color, orientation='vertical')
    plot_hist(y_data, data_range=y_range, truth=y_truth, bins=bins, quantiles=quantiles,
              ax=fig.axes[3], color=color, truth_color=truth_color, orientation='horizontal')
    
    if finalize_fig:
        finalize_figure_for_corner2d(fig, x_label, y_label)
    return fig


def plot_theories_z12(thetas, ranges=None, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    if ranges is None:
        ranges = [None] * len(thetas)
    for th, rg in zip(thetas, ranges):
        if rg is None:
            ax.axline((0.,0.), slope=np.tan(th), **kwargs)
        else:
            z1 = np.multiply(rg, np.cos(th))
            z2 = np.multiply(rg, np.sin(th))
            ax.plot(z1, z2, **kwargs)
    return ax

def plot_theories_spectral(angs, ranges=None, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    if ranges is None:
        ranges = [None] * len(angs)
    for th,rg in zip(angs, ranges):
        if rg is None:
            ax.axvline(th, **kwargs)
        else:
            ax.plot((th,th), rg, **kwargs)
    return ax

def annotate_theories_z12(thetas, labels, label_transformer=lambda x:x,
                          anchors=1., locs=[0.,0.], ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    if np.ndim(anchors) == 0:
        anchors = np.tile(anchors, len(thetas))
    if np.ndim(locs) == 1:
        locs = np.tile(locs, (len(thetas),1))
    for th, lb, ac, lc in zip(thetas, labels, anchors, locs):
        z1 = ac * np.cos(th) + lc[0]
        z2 = ac * np.sin(th) + lc[1]
        rot = np.mod(th + np.pi/2, np.pi) - np.pi/2
        rot = rot / np.pi * 180
        kwargs.update(dict(rotation=rot, rotation_mode='anchor'))
        ax.text(z1, z2, label_transformer(lb), **kwargs)
    return ax

def annotate_theories_spectral(angs, labels, label_transformer=lambda x:x,
                               anchors=1., locs=[0.,0.], ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    if np.ndim(anchors) == 0:
        anchors = np.tile(anchors, len(angs))
    if np.ndim(locs) == 1:
        locs = np.tile(locs, (len(angs),1))
    for th, lb, ac, lc in zip(angs, labels, anchors, locs):
        kwargs.update(dict(rotation='vertical', rotation_mode='anchor'))
        ax.text(th, ac, label_transformer(lb), **kwargs)
    return ax

def get_pn_label_from_ppe_index(b):
    twice_pn = int(b) + 5
    if twice_pn % 2:
        label = f'${twice_pn/2:0.1f}$PN'
    else:
        label = f'${twice_pn//2:d}$PN'
    return label

def get_lines_for_legend(colors, linestyles=None):
    from matplotlib.lines import Line2D
    lines = []
    if linestyles is None:
        linestyles = [None] * len(colors)
    for c,sty in zip(colors, linestyles):
        lines.append(Line2D([0], [0], color=c, linestyle=sty))
    return lines
        

def get_path_unit_square_minus_circle_righthalf(ratio=0.8):
    from matplotlib.path import Path
    path = Path.unit_circle_righthalf()
    vt = np.concatenate(
            [path.vertices[:-1] * ratio, 
             [[0.,1.], [1.,1.], [1.,-1.], [0.,-1.]], 
             path.vertices[[-1]] * ratio,])
    cd = np.concatenate(
            [path.codes[:-1], 
             [2, 2, 2, 2, 79],])
    return Path(vt, cd)


def plot_square_minus_circle_noedge(center=(0.,0.), width=2., ratio=0.8, ax=None, color=None, **kwargs):
    from matplotlib.patches import PathPatch
    from matplotlib.transforms import Affine2D
    if ax is None:
        ax = plt.gca()
    path = get_path_unit_square_minus_circle_righthalf(ratio)
    path = path.transformed(Affine2D().scale(width/2.).translate(*center))
    if color is None:
        color = PathPatch(path, color=color).get_facecolor()
    patch = PathPatch(path, color=color, **kwargs)
    ax.add_patch(patch)
    path = get_path_unit_square_minus_circle_righthalf(ratio)
    path = path.transformed(Affine2D().rotate(np.pi))
    path = path.transformed(Affine2D().scale(width/2.).translate(*center))
    patch = PathPatch(path, color=color, **kwargs)
    ax.add_patch(patch)
    return ax
