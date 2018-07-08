import numpy as np


class Board():
    """
    """
    def __init__(self, n, hole):
        self.board = np.zeros((2**n, 2**n))
        self.board[hole] = 1
        self.populate()

    def populate(self):
        """
        """

        x0, x1 = 
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
