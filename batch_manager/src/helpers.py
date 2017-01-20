from pyparsing import ParseResults


class Tree(object):
    """
    Tree for evaluating nested function calls and arguments

    :param pyparsing.ParseResults parsed_args: parsed computation
    """
    def __init__(self, parsed_args):
        self.root = Node(parsed_args, None)

    def collect_leaves(self, node=None, leaves=None):
        """
        Collect leaves of tree

        :param Node node:  node in tree
        :param set leaves: collected leaves
        :return: collected leaves
        :rtype: set
        """
        if not leaves:
            leaves = set([])
        if not node:
            node = self.root

        if not node.is_leaf():
            for child in node.children:
                if isinstance(child, Node):
                    leaves = self.collect_leaves(child, leaves)
        else:
            leaf = tuple([node.f, tuple(node.children)])
            leaves.add(leaf)
        return leaves

    def prune(self, known, node=None):
        """
        Prune tree with evaluating Nodes by information in known\
                expanded nodes have Node.val attribute assigned

        :param defaultdict.str.tuple known: map for returns of functions
        :param Node node: current location
        """
        if not node:
            node = self.root
        for child in node.children:
            if isinstance(child, Node):
                self.prune(known, child)

        # Empty argument case, these can be combined by
        # Modifying types in arg parse !!!TODO
        if all([not node.children, node.is_leaf(), () in known[node.f]]):
            node.val = known[node.f][()]
            if node.parent:
                node.parent.children[:] = [i.val if isinstance(i, Node) and
                                           i.val is not None else i
                                           for i in node.parent.children]
        elif node.is_leaf() and tuple(int(i) for i in node.children)\
                in known[node.f]:
            node.val = known[node.f][tuple(int(i) for i in node.children)]
            if node.parent:
                node.parent.children[:] = [i.val if isinstance(i, Node) and
                                           i.val is not None else i for
                                           i in node.parent.children]


class Node(object):
    """
    Node for storing a parsed computation function and argument

    :param pyparsing.ParseResults parsed_args: parsed computation
    :param Node parent: reference to parent node
    """
    def __init__(self, parsed_args, parent=None):
        f, args = parsed_args[0], parsed_args[1]
        self.f = f
        self.parent = parent
        self.children = []
        self.val = None
        for i in args:
            if isinstance(i, ParseResults):
                self.children.append(Node(i, self))
            else:
                try:
                    self.children.append(int(i))
                except ValueError:
                    pass

    def is_leaf(self):
        """
        Check if node is a leaf in tree, none of it's children are type(Node)

        :return: if  node is leaf
        :rtype: bool
        """
        return not any([isinstance(child, Node) for child in self.children])


def s_sum(lst):
    """
    Sum list of string integers

    :param list lst: list of integers as type strings
    :return: sum
    :rtype: int
    """
    return sum([int(i) for i in lst])


def s_max(lst):
    """
    Sum list of string integers

    :param list lst: list of integers as type strings
    :return: max
    :rtype: int
    """
    return max([int(i) for i in lst])


def ldiv(lst):
    """
    len(lst) == 2 division lst[0]/lst[1]

    :param list lst: length 2 list of divisbles
    :return: quotient
    :rtype: argmax(lst).dtype
    """
    lst = [int(i) for i in lst]
    return lst[0] / lst[1]


def batch_sum(lstlst):
    """
    batch sum function, sum each sublists
    """
    return [s_sum(i) for i in lstlst]


def batch_max(lstlst):
    """
    batch max function, max each sublists
    """
    return [s_max(i) for i in lstlst]


def batch_ldiv(lstlst):
    """
    batch ldiv function, ldiv each sublists
    """
    return [ldiv(i) for i in lstlst]
