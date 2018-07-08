import unittest


class Register():
    def __init__(self, total, coins):
        self.total = total
        self.coins = coins
        self.tested = set([])
        self.valid = set([])
        self.d = {}
        self.make_change(total, coins)

    def make_change(self, n, coins, used=None):
        if not used:
            used = []
        if n == 0:
            self.valid.add(tuple(used))
        elif n < 0:
            return None
        else:
            for i in coins:
                s = sorted(used + [i])
                t = tuple(s)
                if t not in self.tested:
                    self.tested.add(t)
                    self.make_change(n-i, coins, s)


class TestX(unittest.TestCase):

    def test_x(self):
        pass

if __name__ == "__main__":
    unittest.main()
