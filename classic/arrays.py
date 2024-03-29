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
