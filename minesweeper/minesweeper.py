import pandas as pd
import itertools as it
import random


class MineSweeper(object):
    """
    Class for owning minesweeper game. Provides interaction wtih :py:class:`Board`

    :param tuple.int size: 1x2 tuple of board size (r, c)
    :param int numBombs: number of mines in the game
    :param str directions: 'octile' - 8 way connected, 'quad' 4 way connected
    :param list.tuple.int: option to specify bomb locations, default selected random
    """
    def __init__(self, size=(10, 10), numBombs=10, directions='octile', bombs=None):
        self._size = size
        self.lost = False
        self.won = False

        if directions == 'octile':
            directions = set(it.product([-1, 0, 1], [-1, 0, 1]))
            directions.remove((0, 0))
        self._directions = directions

        self._board = Board(size, directions)
        self._build_bombs(numBombs, bombs)

    @property
    def board(self):
        """
        :py:class:`Board`
        """
        return self._board

    @property
    def bombs(self):
        """
        Bomb locations

        list.tuple.ints
        """
        return self._bombs

    @bombs.setter
    def bombs(self, values):
        self._bombs = values

    def _build_bombs(self, numBombs, bombs):
        if not bombs:
            self._numBombs = numBombs
            self.bombs = random.sample(self.board.cells.keys(), self._numBombs)
        else:
            self._numBombs = len(bombs)
            self.bombs = bombs

        self.board._place_bombs(self.bombs)
        self.board._build_values()

    def _select(self, cell):
        if cell in self.bombs:
            self.lost = True
            return

        else:
            self.board._expose(cell)
            r, c = cell
            if not self.board.df[c][r]:
                self._bfs(cell)

    def _bfs(self, selectedCell):
        for cell in self.board.cells[selectedCell]:
            print cell
            if cell in self.board.closed:
                self.board.closed.remove(cell)
                self.board.opened.add(cell)
                self.board._expose(cell)
                r, c = cell
                if not self.board.df[c][r]:
                    self._bfs(cell)


class Board(object):
    def __init__(self, size, directions):
        self._build_attributes(size, directions)

    @property
    def df(self):
        """
        :py:class:`pandas.DataFrame` for placing bombs and building adjacency
        """
        return self._df

    @property
    def boardState(self):
        """
        Board state visible to :py:class:`Player`.
        """
        return self._boardState

    @property
    def directions(self):
        """
        prescribed directions ('octile', 'quad')
        """
        return self._directions

    @property
    def cells(self):
        """
        Adjacency look up dictionary
        """

        return self._cells

    @property
    def bombs(self):
        """
        Bomb locations

        list.tuple.ints
        """
        return self._bombs

    @property
    def values(self):
        """
        Number of adjacent bombs for non bombs cells, else -1 
        """
        return self._values

    @property
    def closed(self):
        """
        List of unvisited nodes
        """
        return self._closed

    @property
    def opened(self):
        """
        List of visited nodes
        """

        return self._opened

    def _build_attributes(self, size, directions):
        index = range(size[0])
        columns = range(size[1])

        self._directions = directions
        self._df = pd.DataFrame(data=0, index=index, columns=columns)
        self._boardState = pd.DataFrame(index=index, columns=columns)

        self._cells = {i: set([]) for i in it.product(index, columns)}
        self._values = {i: 0 for i in it.product(index, columns)}

        self._closed = set(self._cells.keys())
        self._opened = set([])

        self._build_adjacency()

    def _place_bombs(self, bombs):
        """
        """
        self._bombs = bombs
        for r, c in bombs:
            self.df[c][r] = -1

    def _build_adjacency(self):
        """
        """
        for cell in self._cells:
            for i, j in self._directions:
                y = cell[1] + j
                x = cell[0] + i
                if y in self._df and x in self._df[y]:
                    self._cells[cell].add((x, y))

    def _build_values(self):
        """
        """
        for bomb in self.bombs:
            for r, c  in self.cells[bomb]:
                if (r,c)  not in self.bombs:
                    self.df[c][r] += 1

    def _expose(self, cell):
        r, c = cell
        self._boardState[c][r] = self.df[c][r]
