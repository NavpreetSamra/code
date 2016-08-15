import json
import regex


def parse(fileName):
    """
    Parse csv with node id + JSON blob, links are added if they meet:

    * 2.4GHz frequency >= 8 MCS quality
    * 5GHz frequency >= 6 MCS quality

    Note this parser is built to handle a specifically malformed JSON blob where both "{""key"":""val""}" "{""key"":val""}" exist inline

    :param str fileName: Relative path to csv with node id + JSON blob

    :return nodeMap: map from MAC to node id {MAC: Node} ; {str: str}
    :return macMap: map from node id to MACs {Node: [MAC1, MAC2]} ; {str: list.str}
    :return mesh: MAC mesh {MAC: [MAC1, ... MACi]} ; {str, list.str}
    :return links: link quality (NB not bidirectional) from MAC1 to MAC2 {(MAC1, MAC2): val}; {tuple.str, int}
    :return badlinks: links where the rx_bitrate is  <= 11.0 MB/s and did not qualify for MCS index 
    :rtype: tuple.dict
    """
    expression = regex.compile(r"(VHT-MCS )([0-9]+)")

    f = open(fileName, 'r')
    nodeMap = {}
    macMap = {}
    mesh = {}
    links = {}
    bad_links = {}
    for l in f.readlines()[1:]:
        node,  blob = l.strip().split(',', 1)

        blob = blob.replace('\"{', '{')
        blob = blob.replace('{\"', '{')
        blob = blob.replace('\"}', '}')
        blob = blob.replace('}\"', '}')
        blob = blob.replace('\",', ',')
        blob = blob.replace(',\"', ',')
        blob = blob.replace('\":', ':')
        blob = blob.replace(':\"', ':')

        d = json.loads(blob)

        macs = d.keys()
        macMap[int(node)] = macs
        for i in macs:
            nodeMap[i] = int(node)
            macLinks = d[i]['links'].keys()
            mesh[i] = macLinks
            for j in macLinks:
                linkQuality = d[i]['links'][j]['rx_bitrate']
                match = regex.search(expression, linkQuality)
                if match: 
                    rx = int(match.captures(2)[0])
                else:
                    bad_links[(j, i)] = linkQuality
                    rx = 0

                linkValue = (j[-1] == '3' and rx >= 6) or  \
                            (j[-1] == '4' and rx >= 8)
                links[(j, i)] = linkValue

    return nodeMap, macMap, mesh, links, bad_links
