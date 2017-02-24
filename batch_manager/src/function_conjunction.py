from collections import defaultdict, Counter
from pyparsing import (Forward, Word, alphas, alphanums,
                       nums, ZeroOrMore, Literal, Group,
                       ParseResults, Empty, Combine, Optional)
from helpers import Tree


class Compute(object):
    """
    Class for computing composed functions
    """
    @staticmethod
    def _format_args(aStr):
        """
        Process composed function string into nested pyparsing.ParseResults

        :param str aStr: string to parse
        :return: formatting result
        :rtype: pyparsing.ParseResults
        """

        identifier = Word(alphas, alphanums + "_")
        integer = Combine(Optional(Literal('-')) + Word(nums))
        functor = identifier
        lparen = Literal("(").suppress()
        rparen = Literal(")").suppress()
        expression = Forward()
        arg = Group(expression) | identifier | integer | Empty()
        args = arg + ZeroOrMore("," + arg)
        expression << functor + Group(lparen + args + rparen)
        return expression.parseString(aStr)

    @staticmethod
    def select_func(leaves):
        """
        Select function to evaluate and collect arguments

        :param set.tuple.str,list.int leaves: function argument pairs
        :return: FunctionRegistry key
        :rtype: str
        :return: arguments for function to operate on
        :rtype: list.list.str
        """
        count = Counter([i[0] for i in leaves])
        func = count.most_common(1)[0][0]
        argSet = set([tuple(i[1]) for i in leaves if i[0] == func])
        args = [list(i) for i in argSet]
        return func, args

    @classmethod
    def serial(cls, computations):
        """
        Evaluation of each computation in serial

        :param list.str computations: list of string computations to execute
        :return: computation results
        :rtype: list
        """
        output = []
        for computation in computations:
            parsed_args = cls._format_args(computation)
            output.append(cls._serial_eval(parsed_args))
        return output

    @classmethod
    def _serial_eval(cls, parsed_args):
        """
        Evaulate computation specified by parsed_args in serial

        :param pyparsing.ParseResults parsed_args: computation
        :return: function applied on args
        """

        f, args = parsed_args[0], parsed_args[1]
        parsed = []
        for i in args:
            if isinstance(i, ParseResults):
                parsed.append(cls._serial_eval(i))
            else:
                try:
                    if i != ',':
                        parsed.append(i)
                except ValueError:
                    pass
        return cls.funcs[f](parsed)

    @classmethod
    def batch(cls, computations):
        """
        Evaluation of each computation in batch

        :param list.str computations: list of string computations to execute
        :return: computation results
        :rtype: list
        """
        trees = {ind: Tree(cls._format_args(i))
                 for ind, i in enumerate(computations)}
        output = cls._batch_eval(trees)

        if cls.verbose:
            print 'batch calls: ' + str(cls.batchCalls)
        return output

    @classmethod
    def _batch_eval(cls, trees):
        """
        Evaulate computations in each tree containing parsed arguments

        :param dict trees: dictionary of pyparsing.ParseResults computation
        :return: computation results
        :rtype: list
        """
        if all([i.root.val is not None for i in trees.itervalues()]):
            return [trees[i].root.val for i in sorted(trees)]
        subtrees = [i for i in trees.itervalues() if i.root.val is None]
        leaves = set([])

        for tree in subtrees:
            leaves = leaves.union(tree.collect_leaves())

        func, args = cls.select_func(leaves)
        cls._eval_leaves(func, args)

        for tree in subtrees:
            tree.prune(cls.memo)

        return cls._batch_eval(trees)

    @classmethod
    def _eval_leaves(cls, func, args):
        """
        Apply function to batch args, store results in cls.memo

        :param str func: key in FunctionRegistry for function
        :param list.list args: batch args
        """
        cls.batchCalls += 1
        args = [[j for j in i] for i in args]
        batched = cls.funcs[func](args)
        for key, value in zip(args, batched):
            cls.memo[func][tuple(key)] = value

    @classmethod
    def compute(cls, computations,
                functionRegistry,
                evalType='serial', verbose=False):
        """
        Evaluate computations using functions supplied in functionRegistry\
                with either serial or batch methods

        :param list.str computations: computations to evaluate
        :param dict functionRegistry: map strings in computations to functions
        :param str evalType: method for evaluation. serial requires functions\
                in registry to accept a list.int, batch requires functions\
                in registry to accept list.list.int
        :param bool verbose: output additional information during run\
                currently only prints number of batch calls if batch evalType
        :return: evaluated computations
        :rtype: list
        """

        cls.funcs = functionRegistry
        cls.evalType = evalType
        cls.verbose = verbose
        cls.memo = defaultdict(dict)
        cls.batchCalls = 0
        return cls.__dict__[evalType].__func__(cls, computations)


if __name__ == "__main__":
    pass
