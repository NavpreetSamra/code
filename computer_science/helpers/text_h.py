import numpy as np
import regex
import ntpath



def regex_recursive(txt, seps=["\(", "\)"], all_groups=False):
    """
    Regex for nested divider pairs for capturing

    :param str txt: line to parse
    :param array-like seps: list of capturing group
    :param bool all_groups: return top level group(False). all groups (True)

    :return result.capture: group(s) caputred
    :rtype: str | list
    """
    [sep1, sep2] = seps
    result = regex.search(r'''
                            (?<rec>
                             ''' + sep1 + '''
                             (?:
                              [^''' + sep1 + sep2 + '''+)]++
                              |
                               (?&rec)
                             )*
                             ''' + sep2 + '''
                            )
                           ''', txt, flags=regex.VERBOSE)
    if all_groups:
        return result.captures('rec')
    else:
        return result.captures('rec')[-1]


def find_sep(l):
    """
    Find key value seperator in line

    :param str l: line of log

    :return key: seperator type
    :rtype: str
    """

    d = {}
    if '(' in l:
        d['('] = l.index('(')

    if ',' in l:
        d[','] = l.index(',')

    try:
        key = min(d, key=d.get)
    except ValueError:
        key = None
    return key


def find_end_iter(l, sep_dict):
    """
    Find end of iteration based on seperator and return first split accounting for \
            nested separators
    
    :param str l: line to parse
    :param dict sep_dict: dictionary of bounding pair for recursive search

    :return: (l1, l2)
    :rtype: tuple
    """
    exposed_comma = False
    hold = []

    if ',' not in l:
        return l, ""

    while not exposed_comma:
        if sep_dict.key() in l:
            if l.index(sep_dict.key()) > l.index(','):
                l1, l = l.split(', ', 1)
                hold.extend(l1)
                exposed_comma = True
            else:
                hold_value = regex_recursive(l, sep_dict["("])
                value = l.split(hold_value, 1)[0] + hold_value

                hold.extend(value)
                l = l.split(value, 1)[1]
        else:
            exposed_comma = True
    return "".join(hold).rstrip(','), l

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start


def path_leaf(abspath):
    """
    """
    folder, file = ntpath.split(abspath)
    return file or ntpath.basename(file)

