import sys
import numpy as np
import itertools as it
import networkx as nx
import json
sys.setrecursionlimit(1000)


class SystemNetwork(object):
    """
    Class for identifying systems of networks from sets of \
    MAC address peer links

    SystemNetwork takes mappings between nodes via MAC addresses and the link's
    quality and creates clusters of networks by partitioning nodes via
    walking through the mesh. Then SystemNetwork evaluates  \
    each network to ensure each node has a quality link with each other node
    in its network. Networks are attributed \
            True(paths exists with quality links
    to/from each node) or False. If there is only 1 node in a \
            network it is recorded
    as not a network and a warning indicating such is reported.

    :param dict nodeMap: map from MAC to node id {MAC: Node} ; {str: str}
    :param dict macMap: map from node id to MACs {Node: [MAC1, MAC2]} \
            ; {str: list.str}
    :param dict mesh: MAC mesh {MAC: [MAC1, ... MACi]} ; {str, list.str}
    :param dict links: link quality (NB not bidirectional) \
            from MAC1 to MAC2 {(MAC1, MAC2): None}; {tuple.str, int}

    The *file peer_link_example.csv* is an example file which can be loaded
    with the `.example_parser` to generate the inputs for system network
    """
    def __init__(self, nodeMap, macMap, mesh, links):
        self.nodeMap = nodeMap
        self.macMap = macMap
        self.links = links
        self.mesh = mesh

        self.node_list = self.macMap.keys()
        self.mac_list = self.nodeMap.keys()

        # Populated in methods
        self.networks = {}
        self.nodes = {i: [] for i in self.node_list}
        self.nodeLinks = {}
        self.graphs = {}
        self.results = {}

        # Seed for mesh walk
        self._seed = self.node_list[0]
        # Initilize networks for mesh walk
        self.networks = {0: list([self._seed])}
        # Tracker for mesh walk
        self._nodeTrack = np.array(self.node_list[1:])

        # Can be moved to owner or front end depending on model implementation
        self.macs_to_nodes()
        self.partition_system(self._seed)
        self.evaluate_networks()

    def macs_to_nodes(self):
        """
        Convert MAC Links to Node Links
        """
        for node in self.node_list:
            for mac in self.macMap[node]:
                links = self.mesh[mac]
                if not isinstance(links, list):
                    links = list([links])
                for l in links:
                    if self.links[(l, mac)] is True:
                        self.nodeLinks[(self.nodeMap[l], node)] = True

                    # This undirects the graphs to ensure that all associated
                    # links are found in partitioning if graph is not
                    # completely connected. Filtering by connection will
                    # re-instate direction
                    nl = self.nodeMap[l]
                    self.nodes[node].extend([nl])
                    self.nodes[nl].extend([node])

        for node in self.node_list:
            self.nodes[node] = list(set(self.nodes[node]))

    def partition_system(self, new_nodes, index=0):
        """
        Partition mesh into networks

        :param array-like new_nodes: iterable of new_nodes found in crawl
        :param int index: network id (increments via recursion)
        """
        if not isinstance(new_nodes, list):
            new_nodes = list([new_nodes])
        hold = []
        for i in new_nodes:
            hold.extend(self.nodes[i])
        nodes = set(hold)
        if nodes:
            new_nodes = []
            for i in nodes:
                if i not in self.networks[index]:
                    new_nodes.extend(list([i]))
            if any(new_nodes):
                self.networks[index].extend(new_nodes)
                self._nodeTrack = self._nodeTrack[np.in1d(self._nodeTrack,
                                                          np.array(new_nodes),
                                                          invert=True)]
                self.partition_system(new_nodes, index)

        if any(self._nodeTrack):
            index += 1
            seed = self._nodeTrack[0]
            self.networks[index] = list([seed])
            if len(self._nodeTrack) > 1:
                self._nodeTrack = self._nodeTrack[1:]
            else:
                self._nodeTrack = np.array([])
            self.partition_system(seed, index)

    def evaluate_networks(self):
        """
        Create :class:`.MeshGraph` from node lists in networks and evaluate quality
        """

        for i, network in enumerate(self.networks.values()):
            self.graphs[i] = MeshGraph(network, self.nodeLinks)

    def create_json(self, fname='analysis.json'):
        """
        Create json written to fname with results

        :param str fname: name of file to write data out to as json
        """
        out = {}
        for i in self.networks:
            d = {}
            d['Connected'] = self.graphs[i].result
            d['Network'] = self.graphs[i].nodes()
            out[i] = d
        outjson = open(fname, 'w')
        j = json.dumps(out, indent=4)
        print >> outjson, j
        outjson.close()


class MeshGraph(nx.DiGraph):
    """
    Subclass of :py:func:`networkx.DiGraph` to add tracking attributes

    :param array-like network: list of node ids in network
    :param dict nodeLinks: dict of quality links from n1-> n2 \
            {(n1,n2)} : {tuple(int, int)}
    """
    def __init__(self, network, nodeLinks, *args, **kwargs):
        super(MeshGraph, self).__init__(*args, **kwargs)
        self.network = network

        self.add_nodes_from(self.network)
        if len(self.network) < 2:
            self._result = "Single node, not a network"
        else:
            self.nodeLinks = nodeLinks

            self._result = None

            self.build_graph()
            self.check_graph()

    @property
    def result(self):
        """
        Boolean of whether network has quality paths from
        all nodes to all other nodes
        """
        return self._result

    def build_graph(self):
        """
        Build directed graph for a network and evaulate its quality
        """
        connections = it.permutations(self.network, 2)
        for i in connections:
            if i in self.nodeLinks:
                if self.nodeLinks[i]:
                    self.add_edge(i[0], i[1])

    def check_graph(self):
        self._result = True
        for i in it.permutations(self.nodes(), 2):
            if not nx.has_path(self, i[0], i[1]):
                self._result = False
                break
