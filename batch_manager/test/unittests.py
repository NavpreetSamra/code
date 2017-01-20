import unittest
from src import function_conjunction as fc
from src.helpers import ldiv, batch_sum, batch_max, batch_ldiv, s_sum, s_max


class TestSerialFuncConj(unittest.TestCase):

    functionRegistry = {'f': s_sum, 'g': s_sum, 'h': s_max}
    evalType = 'serial'

    def test_kwarg(self):
        computations = ['f(g(h(2,3),5),g(g(3),h(4)),10)']
        result = fc.Compute.compute(computations, self.functionRegistry, self.evalType)
        self.assertEqual(result, [25])

    def test_multi(self):
        computations = ['f(g(h(2,3),5),g(g(3),h(4)),10)', 'h(1,f(1,1,2),2)']
        result = fc.Compute.compute(computations, self.functionRegistry, self.evalType)
        self.assertEqual(result, [25, 4])

    def test_zeros(self):
        computations = ['f()', 'g()']
        result = fc.Compute.compute(computations, self.functionRegistry, self.evalType)
        self.assertEqual(result, [0, 0])

    def test_nested_zero(self):
        computations = ['f(g())']
        result = fc.Compute.compute(computations, self.functionRegistry, self.evalType)
        self.assertEqual(result, [0])

    def test_nested_zeros(self):
        computations = ['f(g())', 'g(f())']
        result = fc.Compute.compute(computations, self.functionRegistry, self.evalType)
        self.assertEqual(result, [0, 0])

    def test_order(self):
        computations = ['f(1,2)', 'f(2,1)', 'f(2,f(4,2))']
        functionRegistry = {'f': ldiv}
        result = fc.Compute.compute(computations, functionRegistry, self.evalType)
        self.assertEqual(result, [0, 2, 1])


class TestBatchFuncConj(unittest.TestCase):

    functionRegistry = {'f': batch_sum, 'g': batch_sum, 'h': batch_max}
    evalType = 'batch'

    def test_kwarg(self):
        computations = ['f(g(h(2,3),5),g(g(3),h(4)),10)']
        result = fc.Compute.compute(computations, self.functionRegistry, self.evalType)
        self.assertEqual(result, [25])

    def test_multi(self):
        computations = ['f(g(h(2,3),5),g(g(3),h(4)),10)', 'h(1,f(1,1,2),2)']
        result = fc.Compute.compute(computations, self.functionRegistry, self.evalType)
        self.assertEqual(result, [25, 4])

    def test_zeros(self):
        computations = ['f()', 'g()']
        result = fc.Compute.compute(computations, self.functionRegistry, self.evalType)
        self.assertEqual(result, [0, 0])

    def test_nested_zero(self):
        computations = ['f(g())']
        result = fc.Compute.compute(computations, self.functionRegistry, self.evalType)
        self.assertEqual(result, [0])

    def test_nested_zeros(self):
        computations = ['f(g())', 'g(f())']
        result = fc.Compute.compute(computations, self.functionRegistry, self.evalType)
        self.assertEqual(result, [0, 0])

    def test_order(self):
        computations = ['f(1,2)', 'f(2,1)', 'f(2,f(4,2))']
        functionRegistry = {'f': batch_ldiv}
        result = fc.Compute.compute(computations, functionRegistry, self.evalType)
        self.assertEqual(result, [0, 2, 1])

    def test_calls1(self):
        computations = ['f(f(0))']
        fc.Compute.compute(computations, self.functionRegistry, self.evalType)
        self.assertEqual(fc.Compute.batchCalls, 1)

    def test_calls2(self):
        computations = ['f(f())']
        fc.Compute.compute(computations, self.functionRegistry, self.evalType)
        self.assertEqual(fc.Compute.batchCalls, 2)

    def test_calls_nest(self):
        computations = ['f(f(f(f(0))))', 'f(f(0))']
        fc.Compute.compute(computations, self.functionRegistry, self.evalType)
        self.assertEqual(fc.Compute.batchCalls, 1)

if __name__ == "__main__":
    unittest.main()
