import unittest


class Sorters(object):
    def merge_sort(self, lst):
        """
        Sort lst with merge sort

        :param list lst: lst of numbers to sort
        """
        n = len(lst)
        if n < 2:
            return lst
        a = lst[:n / 2]
        b = lst[n / 2:]
        a = self.merge_sort(a)
        b = self.merge_sort(b)
        return self.reduce_list_rec(a, b)

    def reduce_list_rec(self, a, b):
        """
        Recursively reduce merge sort components

        :param list a: first component
        :param list b: second component
        """
        if not a:
            return b
        if not b:
            return a

        if a[0] >= b[0]:
            return [b.pop(0)] + self.reduce_list_rec(a, b)
        else:
            return [a.pop(0)] + self.reduce_list_rec(a, b)

    def reduce_list_iter(self, a, b):
        """
        Iteratively reduce merge sort components

        :param list a: first component
        :param list b: second component
        """
        out = []
        while all([a, b]):
            if a[0] > b[0]:
                out.append(b.pop(0))
            else:
                out.append(a.pop(0))
        if not a:
            out.extend(b)
        if not b:
            out.extend(a)

        return out

    def reduce_list_rec_out(self, a, b, out=[]):
        """
        Recursively reduce merge sort components passing ouptut throughout

        :param list a: first component
        :param list b: second component
        """
        if not a:
            out.extend(b)
            return out
        if not b:
            out.extend(b)
            return out

        if a[0] > b[0]:
            out.append(b.pop(0))
            self.reduce_list_rec_out(a, b, out)
        else:
            out.append(a.pop(0))
            self.reduce_list_rec_out(a, b, out)
        return out

    def bubble_sort(self, lst):
        """
        Bubble sort lst

        :param list lst: lst to sort
        """
        flag = True
        n = len(lst)
        if n < 2:
            return lst
        while flag:
            flag = False
            for i in range(n-1):
                if lst[i] > lst[i+1]:
                    lst[i], lst[i+1] = lst[i+1], lst[i]
                    flag = True
        return lst

    def quick_sort(self, lst):
        """
        Quick sort lst

        :param list lst: list to sort
        """
        n = len(lst)
        if n < 2:
            return lst
        pivot, lst = lst[-1], lst[:-1]
        right, left = [], []
        for i in lst:
            if i <= pivot:
                left.append(i)
            else:
                right.append(i)
        return self.quick_sort(left) + list([pivot]) + self.quick_sort(right)


class TestSort(unittest.TestCase):
    """
    Class for testing sorting algorithims
    """
    def test_sorters(self):
        s = Sorters()
        sorts = [s.merge_sort, s.bubble_sort, s.quick_sort]
        for f in sorts:
            print f
            self.assertEqual(f([]), [])
            self.assertEqual(f([1]), [1])
            self.assertEqual(f([1, 2]), [1, 2])
            self.assertEqual(f([2, 1]), [1, 2])
            self.assertEqual(f([1, 2, 3]), [1, 2, 3])
            self.assertEqual(f([3, 1, 2]), [1, 2, 3])

if __name__ == "__main__":
    unittest.main()
