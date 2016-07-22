import sys
import numpy as np
import itertools as it
import networkx as nx
import json
sys.setrecursionlimit(1000)


class SystemNetwork(object):
    """
    Class for identifying systems of networks from sets of MAC address peer links

    :param dict nodeMap: map from MAC to node id {MAC: Node} ; {str: str}
    :param dict macMap: map from node id to MACs {Node: [Mac1, Mac2]} ; {str: list.str}
    :param dict mesh: MAC mesh {MAC: [MAC1, ... Maci]} ; {str, list.str}
    :param dict links: link quality (NB not bidirectional) from Mac1 to Mac2 {(Mac1, Mac2): val}; {tuple.str, int}
    """
    def __init__(self, nodeMap, macMap, mesh, links):
        self.nodeMap = nodeMap
        self.macMap = macMap
        self.links = links
        self.mesh = mesh

        self.node_list = self.macMap.keys()
        self.mac_list = self.nodeMap.keys()

        self.networks = {}
        self.nodes = {i:[] for i in self.node_list}
        self.nodeLinks = {}
        self.graphs = {}
        self.results = {}

        self._seed = self.node_list[0]
        self.networks = {0: list([self._seed])}
        self._nodeTrack = np.array(self.node_list[1:])
  
        self.macs_to_nodes()
        self.partition_system(self._seed)
        self.eval_networks()

    def macs_to_nodes(self):
        """
        Convert Mac Links to Node Links
        """
        for node in self.node_list:
            for mac in self.macMap[node]:
                links = self.mesh[mac]
                if not isinstance(links, list):
                    links = list([links])
                for l in links:
                    if l[-1] == '4' and self.links[(l, mac)] >= 8:
                        self.nodeLinks[(self.nodeMap[l], node)] = True
                    elif l[-1] == '3' and self.links[(l, mac)] >= 6:
                        self.nodeLinks[(self.nodeMap[l], node)] = True
                    # This undirects the graphs to ensure that all associated ...
                    # links are found in partitioning if graph isn't completely connected
                    # Filtering by connection will re-instate direction
                    nl = self.nodeMap[l]
                    self.nodes[node].extend([nl])
                    self.nodes
                    self.nodes[nl].extend([node])
        for node in self.node_list:
            self.nodes[node] = list(set(self.nodes[node]))

    def partition_system(self, new_nodes, index=0):
        """
        Partition mesh into networks 

        :param int i: network id (increments via recursion)
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
                self._nodeTrack = self._nodeTrack[np.in1d(self._nodeTrack, np.asarray(new_nodes), invert=True)]
                self.partition_system(new_nodes, index)
            elif any(self._nodeTrack):
                index += 1
                seed = self._nodeTrack[0]
                self.networks[index] = list([seed])
                if len(self._nodeTrack) > 1:
                    self._nodeTrack = self._nodeTrack[1:]
                else:
                    self._nodeTrack = np.array([])
                self.partition_system(seed, index)

        elif any(self._nodeTrack):
            index += 1
            seed = self._nodeTrack[0]
            self.networks[index] = list([seed])
            if len(self._nodeTrack) > 1:
                self._nodeTrack = self._nodeTrack[1:]
            else:
                self._nodeTrack = np.array([])
            self.partition_system(seed, index)
    
    def eval_networks(self):
        """
        Evaluate quality of networks found in partition_system
        """
        for i in self.networks:
            ntw = self.networks[i]
            if len(ntw) > 1:
                self.results[i] = self.build_graph(i, ntw)
            else:
                self.results[i] = 'Single node network: has no Peers'

    def build_graph(self, index, netwrk):
        """
        Build directed graph for a network

        :param array-like netwrk: array of nodes in network
        :param int index: index of network in system
        :return connected: whether or not all nodes are connected to all other nodes with quality connections
        :rtype: bool
        """
        self.graphs[index] = nx.DiGraph()
        connections = it.permutations(netwrk, 2)
        for i in connections:
            if i in self.nodeLinks:
                if self.nodeLinks[i]:
                    self.graphs[index].add_edge(i[0], i[1])
        connected = True
        for i in it.permutations(netwrk, 2):
            if i[0] not in self.graphs[index] or i[1] not in self.graphs[index]:
                connected = False
                break
            elif not nx.has_path(self.graphs[index], i[0], i[1]):
                connected = False
                break
        return connected

    def create_json(self, fname='analysis.json'):
        out = {}
        for i in self.networks:
            d = {}
            d['Connected'] = self.results[i]
            d['Network'] = self.networks[i]
            out[i] = d
        outjson = open(fname, 'w')
        j = json.dumps(out, indent=4)
        print >> outjson, j
        outjson.close()
