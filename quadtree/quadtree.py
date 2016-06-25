import numpy as np
from helpers import math_h
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm


class QuadTree(object):
    """
    Class for creating quadtree of two dimensional data
    Tree attribute is a dictionary keys are bounding box
    values are coordinates box self.tree[(x1,x2,y1,y2)] = data

    :param array-like data: nx2 array of cartesian points
    :param int max_cell: maximum number of points in a cell
    :param array-like buff: (optional[x,y]) 1x2 add buffer to cells. \
        this includes points in buffer around cell. .25 add 12.5% \
        of cell length to include points outside the cell for functions
        that suffer from edge effects. [.1, .2] adds +/- 5% in x and 10% y
    """
    def __init__(self, data, max_cell, buff=None):

        self.data = np.asarray(data)
        self.max_cell = max_cell
        self.buff = buff

        xs = data[:, 0]
        ys = data[:, 1]

        lx = np.min(xs)
        ux = np.max(xs)
        ly = np.min(ys)
        uy = np.max(ys)

        self.box = np.array([lx, ux, ly, uy])
        self.tree = {}

        self.bounds = {}
        self.bounds[(lx, ux, ly, uy)] = None

        self.parse(self.data, self.box)

    def parse(self, data, box):
        """
        Recursive function for building leaves

        :param array-like data: points to tree
        :param array-like box: [x1, x2, y1, y2] corners of bounding box
        """
        data = math_h.enforce_2d(data)

        if data.shape[0] > self.max_cell:
            xs = data[:, 0]
            ys = data[:, 1]

            [lx, ux, ly, uy] = box
            cx = (lx + ux) / 2.
            cy = (ly + uy) / 2.

            sw_tree = (lx, cx, ly, cy)
            nw_tree = (lx, cx, cy, uy)
            se_tree = (cx, ux, ly, cy)
            ne_tree = (cx, ux, cy, uy)

            sw = np.array(xs <= cx) & np.array(ys <= cy)
            nw = np.array(xs <= cx) & np.array(ys > cy)
            se = np.array(xs > cx) & np.array(ys <= cy)
            ne = np.array(xs > cx) & np.array(ys > cy)

            if sum(sw) <= self.max_cell:
                if self.buff:
                    self.tree[sw_tree] = self.add_buffer(sw_tree)
                else:
                    self.tree[sw_tree] = data[sw, :]
            else:
                self.parse(data[sw, :], np.array(sw_tree))

            if sum(nw) <= self.max_cell:
                if self.buff:
                    self.tree[nw_tree] = self.add_buffer(nw_tree)
                else:
                    self.tree[nw_tree] = data[nw, :]
            else:
                self.parse(data[nw, :], np.array(nw_tree))

            if sum(se) <= self.max_cell:
                if self.buff:
                    self.tree[se_tree] = self.add_buffer(se_tree)
                else:
                    self.tree[se_tree] = data[se, :]
            else:
                self.parse(data[se, :], np.array(se))

            if sum(ne) <= self.max_cell:
                if self.buff:
                    self.tree[ne_tree] = self.add_buffer(ne_tree)
                else:
                    self.tree[ne_tree] = data[ne, :]
            else:
                self.parse(data[ne, :], np.array(ne_tree))

    def add_buffer(self, box):
        """
        Add list of buffer points to buffer attribute of self with shared key to tree

        :param array-like box: [x1, x2, y1, y2] corners of bounding box
        """
        data = math_h.enforce_2d(self.data)
        xs = data[:, 0]
        ys = data[:, 1]

        [lx, ux, ly, uy] = box
        dx = (ux - lx) * self.buff[0] / 2.
        dy = (uy - ly) * self.buff[1] / 2.
        [lx, ux, ly, uy] = [lx - dx, ux + dx, ly - dy, uy + dy]

        logical = np.array(lx < xs) & np.array(ux > xs) & \
                  np.array(ly < ys) & np.array(uy > ys)

        return data[logical, :]


    def plot_outline(self, fig_name='tree_outline.png', points=False):
        """
        Plot bounding boxes (and points optionally)
        """

        corners = np.array(self.tree.keys())
        x1s = corners[:, 0]
        x2s = corners[:, 1]
        y1s = corners[:, 2]
        y2s = corners[:, 3]

        colors = cm.rainbow(np.linspace(0, 1, len(self.tree)))

        if points:
            plt.plot(self.data[:, 0], self.data[:, 1], '.', alpha=.25, c='k')

        for x1, x2, y1, y2, c in zip(x1s, x2s, y1s, y2s, colors):
            plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], c=c)

        plt.savefig(fig_name)
        plt.close()
