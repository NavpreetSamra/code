import json
import regex


def parse(fileName):
    """
    Parse csv with node id + JSON blob
    Note this parser is built to handle a specifically malformed Json blob where both "{""key"":""val""}" "{""key"":val""}" exist inline
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
                links[(j, i)] = rx

    return nodeMap, macMap, mesh, links, bad_links
