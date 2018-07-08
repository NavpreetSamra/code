class Graph():
    def __init__(self, n):
        self.n = n - 1
        self.nodes = {i: set([]) for i in range(n)}
        self._track = set(self.nodes.keys())
        self.pathFrom = {}
        self.pathCost = {}

    def connect(self, n1, n2):
        self.nodes[n1].add(n2)
        self.nodes[n2].add(n1)

    def find_all_distances(self, n):
        self._track.remove(n)
        self.networks = {0: set([n])}
        self.bfs(set([n]))

    def bfs(self, nodes, i=0):
        print nodes, [self.nodes[j] for j in nodes], i, self.networks[i]
        new_nodes = set([])
        for node in nodes:
            [new_nodes.add(j) for j in self.nodes[node]]
        new_nodes = new_nodes - self.networks[i]
        self.networks[i] = self.networks[i].union(new_nodes)
        self._track -= new_nodes
        if new_nodes:
            self.bfs(new_nodes, i)
        else:
            if self._track:
                i += 1
                self.networks[i] = set([self._track.pop()])
                self.bfs(self.networks[i], i)
