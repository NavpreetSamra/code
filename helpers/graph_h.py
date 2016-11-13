import numpy as np
import networkx as nx
import itertools


class TreeGraph(nx.Graph):
    """
    Subclass of networkx Graph for path search applications
    """

    def tree_diameter(self):
        """
        Find diameter node pairs and path in graph
        """

        self.diameter = None

        nodes = self.nodes()
        node_min = np.min(nodes)
        node_max = np.max(nodes)

        edges = []
        [edges.extend(list(i)) for i in self.edges()]
        edge_array = np.asarray(edges).flatten()
        self.counts, self.bins = np.histogram(edge_array,
                                              bins=np.arange(node_min, node_max)
                                              )
        logical = self.counts == 1
        leaves = [int(i) for i in self.bins[logical]]
        iter_paths = itertools.combinations(leaves, 2)

        for tree_path in iter_paths:
            cost = nx.shortest_path_length(self, tree_path[0], tree_path[1])
            if cost > self.diameter:
                self.diameter = cost
                self.diameter_path = nx.shortest_path(self, tree_path[0], tree_path[1])


class Node(object):
    """
    Node structure (not binary restricted) with list of children

    :param hashable value: node value
    :param list history: list of history values, root is None
    """
    def __init__(self, value, history=set([])):
        self.value = value
        self.history = history
        self.children = []

    def add_child(self, childId):
        self.children.append(childId)


class Tree(object):
    """
    """
    def __init__(self, graph):
        self.graph = graph
        self.tree = {}
        self.nodes = set([])

    def _set_root(self, n):
        self.tree[n] = Node(n)
        self.nodeDepth = {n: 0}
        self.depthNode = {0: set([n])}

    def add_child(self, pId, childId):
        """
        """
        self.tree[childId] = Node(childId, self.tree[pId].history.union(set([pId])))
        self.tree[pId].add_child(childId)


class BFSTree(Tree):
    """
    Tree for BFS of Graph

    :param Graph graph: graph of nodes and edges
    """

    def build_n(self, n):
        """
        Build tree from node n in graph

        :param hashable n: id of node n in graph
        """
        self._set_root(n)
        depth = 1
        newNodes = self.walk(n, depth)
        while newNodes:
            depth += 1
            holdNodes = newNodes
            newNodes = set([])
            for node in holdNodes:
                newNodes = newNodes.union(self.walk(node, depth))

    def walk(self, n, depth):
        """
        """
        nodes = self.graph.nodes[n]
        new_nodes = set([])
        for node in nodes:
            if node not in self.tree[n].history:
                new_nodes.add(node)
                self.add_child(n, node)
                if node in self.nodeDepth[node]:
                    self.nodeDepth[node].add(depth)
                else:
                    self.nodeDepth[node].add(depth)
                if depth in self.depthNode:
                    self.depthNode[depth].add(node)
                else:
                    self.depthNode[depth] = set([node])

        return new_nodes


class Graph(object):
    def __init__(self, n1n2weight):
        self.nodes = {}
        self.edges = {}
        for i in n1n2weight:
            n1, n2, weight = i
            if n1 in self.nodes:
                self.nodes[n1].add(n2)
            else:
                self.nodes[n1] = set([n2])
            if n2 in self.nodes:
                self.nodes[n2].add(n1)
            else:
                self.nodes[n2] = set([n1])

            self.edges[tuple(sorted([n1, n2]))] = weight
