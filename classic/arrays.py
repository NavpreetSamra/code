def rotate_clock(lst):
    """
    """
    n = len(lst)
    if n < 2:
        return lst
    return [list(x) for x in zip(*lst[::-1])]


def rotate_counter_clock(lst):
    """
    """
    n = len(lst)
    if n < 2:
        return lst
    return [list(x) for x in zip(*lst)][::-1]


def rotate_clock_inplace(lst):
    """
    """
    if len(lst) < 2:
        return lst

    n = len(lst)
    m = len(lst[0])

    for i in range(m / 2):
        for j in range(i, (n-i-1)):
            lst[i][j], lst[j][m-1-i] = lst[j][m-1-i], lst[i][j]
            lst[i][j], lst[n-1-i][m-1-j] = lst[n-1-i][m-1-j], lst[i][j]
            lst[i][j], lst[n-1-j][i] = lst[n-1-j][i], lst[i][j]

    return lst


def sorted_common_elements(a, b):
    """
    """
    ia = 0
    ib = 0
    out = set([])
    while ia < len(a) or ib < len(b):
        if a[ia] < b[ib]:
            ia += 1
        if a[ia] > b[ib]:
            ia += 1
        if a[ia] == b[ib]:
            out.add(a[ia])
            ia += 1
            ib += 1
    return out


def zero_matrix(m=[[0, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 0]]):
    """
    """
    r, c = set([]), set([])
    for i, row in enumerate(m):
        for j, val in enumerate(row):
            if val == 0:
                r.add(i)
                c.add(j)
    o = []
    for i, row in enumerate(m):
        hold = []
        for j, val in enumerate(row):
            if i not in r and j not in c:
                hold.append(val)
            else:
                hold.append(0)
        o.append(hold)
    return o


class _Heap():
    def __init__(self, lst, direction='max'):
        self.n = len(lst) - 1
        self.arr = lst

        self.direction = direction
        if direction == 'max':
            self.max_heapify(self.n)
        if direction == 'min':
            self.min_heapify()

    def max_heapify(self, k):
        for i in range(k/2, -1, -1):
            if not self.is_heap(i):
                self.sift_down(i)

    def is_heap(self, i):
        return self.arr[i] > max(self.arr[2*i], self.arr[2*i+1])

    def sift_down(self, i):
        if self.arr[2*i] >= self.arr[2*i+1]:
            self.arr[2*i], self.arr[i] = self.arr[i], self.arr[2*i]
            if 2*i <= self.n/2:
                if not self.is_heap(2*i):
                    self.max_heapify(2*i)
        else:
            self.arr[2*i+1], self.arr[i] = self.arr[i], self.arr[2*i+1]
            if 2*i+1 <= self.n/2:
                if not self.is_heap(2*i+1):
                    self.max_heapify(2*i+1)
