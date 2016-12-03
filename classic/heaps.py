import unittest


class Heap():
    """
    Min heap (native python 2.7)

    :param list lst: array of to heap
    """
    def __init__(self, lst):
        self.n = len(lst)
        self.arr = [0] + lst
        self.heapify()

    def is_empty(self):
        """
        :return: if heap is empty
        :rtype: bool
        """
        return self.n < 1

    def peek(self):
        """
        :return: min value of heap
        :rtype: lst.i
        """
        return self.arr[1]

    def pop(self):
        """
        :return: min value of heap and remove from heap
        :rtype: lst.i
        """
        self.n -= 1
        value = self.arr.pop(1)
        self.heapify()
        return value

    def heapify(self):
        """
        heapify
        """
        for i in range(self.n/2, 0, -1):
            if not self.is_heap(i):
                self.sift_down(i)

    def is_heap(self, i):
        """
        :return: if node + children is empty
        :rtype: bool
        """
        return self.arr[i] < self.arr[2*i + _minarg(self.arr[2*i:2*i+2])]

    def sift_down(self, i):
        """
        bubble down heap from node i

        :param int i: index of lst to bubble
        """

        if not self.is_heap(i):
            ind = _minarg(self.arr[2*i:2*i+2])
            self.swap(i, 2*i + ind) 
        if 2*i+1 <= self.n/2:
            self.sift_down(2*i + ind)

    def swap(self, i, j):
        """
        Swap i <-> j elements of self.arr

        :param int i: index of ith element
        :param int j: index of jth element
        """
        self.arr[i], self.arr[j] = self.arr[j], self.arr[i]

    def insert(self, val):
        """
        Insert val into heap

        :param obj val: val to insert in heap
        """
        self.n += 1
        self.arr += [val]
        self.heapify()


def _minarg(lst):
    """
    Argmin from lst, with handling for None values

    :param list lst: list to argmin
    """
    return [i[0] for i in sorted(enumerate(lst), key=lambda x:x[1]) if i[1]][0]


class TestBrackets(unittest.TestCase):
    def test_sort(self):
        heap = Heap(range(1, 9))
        slist = [heap.pop() for _ in range(heap.n)]
        self.assertEqual(slist, range(1, 9))

    def test_empty(self):
        heap = Heap([])
        self.assertTrue(heap.is_empty())

    def test_build(self):
        heap = Heap([])
        [heap.insert(i) for i in range(1, 9)[::-1]]
        slist = [heap.pop() for _ in range(heap.n)]
        self.assertEqual(slist, range(1, 9))


if __name__ == "__main__":
    unittest.main()
