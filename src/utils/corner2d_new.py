import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import corner

from .defaults import get_corner_default
# from .defaults import get_corner_default_bilby


def find_canonical_range(data, data_range=None):
    if data_range is None:
        data_range = [np.min(data), np.max(data)]
    elif type(data_range) == float:
        data_range = np.quantile(data, [(1-data_range)/2, (1+data_range)/2])
    return data_range

def select_by_keys(d, keys=None):
    if keys is not None:
        d = {k:d[k] for k in set(keys)}
    return d

def plot_hist2d(
        x_data, y_data, 
        x_range=None, y_range=None,
        bins=30, levels=[0.5,0.9], 
        ax=None, color='C0'):
    if ax is None:
        ax = plt.gca()
    x_range = find_canonical_range(x_data, x_range)
    y_range = find_canonical_range(y_data, y_range)
    kwargs = get_corner_default(color=color, bins=bins, levels=levels)
    corner.hist2d(x_data, y_data, range=[x_range,y_range], ax=ax, **kwargs)
    return ax

def plot_hist(
        data, data_range=None,
        bins=30, quantiles=[0.05,0.95],
        ax=None, color='C0', orientation='vertical'):
    if ax is None:
        ax = plt.gca()
    data_range = find_canonical_range(data, data_range)
    bins = np.linspace(*data_range, bins+1)
    ax.hist(data, bins=bins, density=True, 
            color=color, histtype='step', orientation=orientation)
    l, r = np.quantile(data, quantiles)
    plot_line = ax.axvline if orientation == 'vertical' else ax.axhline
    plot_line(l, c=color, linestyle='dashed')
    plot_line(r, c=color, linestyle='dashed')
    return ax

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

def plot_theories_zspectral(angs, ranges=None, ax=None, **kwargs):
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

def annotate_theories_zspectral(angs, labels, label_transformer=lambda x:x,
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

def get_pn_label_from_ppe_index(b):
    twice_pn = int(b) + 5
    if twice_pn % 2:
        label = f'${twice_pn/2:0.1f}$PN'
    else:
        label = f'${twice_pn//2:d}$PN'
    return label

def get_lines_for_legend(colors, linestyles=None, markers=None):
    from matplotlib.lines import Line2D
    lines = []
    if linestyles is None:
        linestyles = [None] * len(colors)
    if markers is None:
        markers = [None] * len(colors)
    for c,sty,mkr in zip(colors, linestyles, markers):
        lines.append(Line2D([0], [0], color=c, linestyle=sty, marker=mkr))
    return lines



class Corner2D(object):

    def __init__(self, fig=None, subplots_ratio=5., subplots_space=0.05):
        if fig is None:
            fig = plt.figure()
        fig.subplots(2, 2, 
                     width_ratios=[subplots_ratio, 1.],
                     height_ratios=[1., subplots_ratio])
        fig.subplots_adjust(wspace=subplots_space, hspace=subplots_space)
        fig.axes[1].axis('off')
        fig.axes[0].set_yticks([])
        fig.axes[3].set_xticks([])
        fig.axes[0].set_xticklabels([])
        fig.axes[3].set_yticklabels([])
        self.figure = fig
        self.axes = fig.axes

    def get_xyplot(self):
        return self.axes[2]
    def get_xplot(self):
        return self.axes[0]
    def get_yplot(self):
        return self.axes[3]

    def set_xlim(self, left=None, right=None, **kwargs):
        self.axes[2].set_xlim(left, right, **kwargs)
        self.axes[0].set_xlim(self.figure.axes[2].get_xlim())
    def set_xticks(self, ticks, labels=None, **kwargs):
        self.axes[2].set_xticks(ticks, labels, **kwargs)
        self.axes[0].set_xticks(ticks, **kwargs)
    def set_xticklabels(labels, **kwargs):
        self.axes[2].set_xticklabels(labels, **kwargs)

    def set_ylim(self, bottom=None, top=None, **kwargs):
        self.axes[2].set_ylim(bottom, top, **kwargs)
        self.axes[3].set_ylim(self.figure.axes[2].get_ylim())
    def set_yticks(self, ticks, labels=None, **kwargs):
        self.axes[2].set_yticks(ticks, labels, **kwargs)
        self.axes[3].set_yticks(ticks, **kwargs)
    def set_yticklabels(labels, **kwargs):
        self.axes[2].set_yticklabels(labels, **kwargs)

    def align_axes(self):
        for minor in (False, True):
            xticks = self.axes[2].get_xticks(minor=minor)
            yticks = self.axes[2].get_yticks(minor=minor)
            self.axes[0].set_xticks(xticks, minor=minor)
            self.axes[3].set_yticks(yticks, minor=minor)
        xlim = self.axes[2].get_xlim()
        ylim = self.axes[2].get_ylim()
        self.set_xlim(xlim)
        self.set_ylim(ylim)

    def rescale_axes(self):
        self.axes[2].autoscale()
        self.align_axes()

    def plot(self, x_data, y_data, 
             x_range=None, y_range=None, 
             x_truth=None, y_truth=None,
             bins=30, levels=[0.5,0.9], quantiles=[0.05,0.95],
             color='C0', truth_color='black', 
             truth_marker='s', truth_markersize=None):
        plot_hist2d(x_data, y_data, 
                    x_range=x_range, y_range=y_range,
                    bins=bins, levels=levels, 
                    ax=self.axes[2], color=color)
        plot_hist(x_data, data_range=x_range,
                  bins=bins, quantiles=quantiles, 
                  ax=self.axes[0], color=color, orientation='vertical')
        plot_hist(y_data, data_range=y_range,
                  bins=bins, quantiles=quantiles, 
                  ax=self.axes[3], color=color, orientation='horizontal')
        self.mark(x_truth, y_truth, color=truth_color, 
                  marker=truth_marker, markersize=truth_markersize)

    def mark(self, x=None, y=None, 
             color='black', linestyle='solid',
             marker='s', markersize=None, **kwargs):
        if x is not None:
            self.axes[2].axvline(x, c=color, linestyle=linestyle, **kwargs)
            self.axes[0].axvline(x, c=color, linestyle=linestyle, **kwargs)
        if y is not None:
            self.axes[2].axhline(y, c=color, linestyle=linestyle, **kwargs)
            self.axes[3].axhline(y, c=color, linestyle=linestyle, **kwargs)
        if x is not None and y is not None:
            self.axes[2].plot(x, y, c=color, marker=marker, markersize=markersize, **kwargs)
        return self



class CornerZ12(Corner2D):

    def __init__(self, fig=None, zmax=1., 
                 grid_indices=[], grid_angles=[], **kwargs):
        super().__init__(fig, **kwargs)
        self.axes[2].set_xlabel(r'$z_1$')
        self.axes[2].set_ylabel(r'$z_2$')
        self.grid = {k:v for k,v in zip(grid_indices, grid_angles)}
        self.zmax = zmax

    def equalize_ranges(self):
        xlim = self.axes[2].get_xlim()
        ylim = self.axes[2].get_ylim()
        xcent = 0.5 * (xlim[1]+xlim[0])
        ycent = 0.5 * (ylim[1]+ylim[0])
        width = max(xlim[1]-xlim[0], ylim[1]-ylim[0])
        xlim = xcent-width/2, xcent+width/2
        ylim = ycent-width/2, ycent+width/2
        self.set_xlim(xlim)
        self.set_ylim(ylim)

    def align_axes(self):
        self.equalize_ranges()
        super().align_axes()

    def add_shade(self, color='lightgray', **kwargs):
        plot_square_minus_circle_noedge(
            width=self.zmax*4., ratio=0.5, 
            ax=self.axes[2], color=color, **kwargs)

    def add_ppe_grid(self, indices=None, color='gray', linestyle='dotted', **kwargs):
        grid = select_by_keys(self.grid, indices)
        plot_theories_z12(
            grid.values(), ax=self.axes[2], 
            color=color, linestyle=linestyle, **kwargs)

    def add_ppe_results(self, indices, cis, color='C0', **kwargs):
        grid = select_by_keys(self.grid, indices)
        plot_theories_z12(
            grid.values(), cis, ax=self.axes[2], 
            color=color, **kwargs)

    def add_ppe_annotations(self, indices, anchors=1., **kwargs):
        grid = select_by_keys(self.grid, indices)
        annotate_theories_z12(
            grid.values(), grid.keys(), 
            label_transformer=get_pn_label_from_ppe_index,
            ax=self.axes[2], anchors=anchors, va='baseline', ha='right', **kwargs)



class CornerZSpectral(Corner2D):

    def __init__(self, fig=None, 
                 grid_indices=[], grid_angles=[], **kwargs):
        super().__init__(fig, **kwargs)
        self.axes[2].set_xlabel(r'$\varphi$')
        self.axes[2].set_ylabel(r'$z_b$')
        self.grid = {k:v for k,v in zip(grid_indices, grid_angles)}

    def add_ppe_grid(self, indices=None, color='gray', linestyle='dotted', **kwargs):
        grid = select_by_keys(self.grid, indices)
        plot_theories_zspectral(
            grid.values(), ax=self.axes[2], 
            color=color, linestyle=linestyle, **kwargs)
        plot_theories_zspectral(
            grid.values(), ax=self.axes[0], 
            color=color, linestyle=linestyle, **kwargs)

    def add_ppe_results(self, indices, cis, color='C0', **kwargs):
        grid = select_by_keys(self.grid, indices)
        plot_theories_zspectral(
            grid.values(), cis, ax=self.axes[2], 
            color=color, **kwargs)

    def add_ppe_annotations(self, indices, anchors=1., **kwargs):
        grid = select_by_keys(self.grid, indices)
        annotate_theories_zspectral(
            grid.values(), grid.keys(), 
            label_transformer=get_pn_label_from_ppe_index,
            ax=self.axes[2], anchors=anchors, va='baseline', ha='right', **kwargs)

