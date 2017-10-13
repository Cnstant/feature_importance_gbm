import re


def tree_cleaning(tree_raw):
    """
    Remove the caracters \t & \n in the initial dump and generate the tree as a list of nodes
    :param tree_raw: tree as raw text
    :return: tree as a list of nodes
    :rtype: list of str
    """
    return tree_raw.replace('\t', '').split('\n')[:-1]


def node_information(string):
    """
    Remove the caracters \t & \n in the initial dump and generate the tree as a list of nodes
    :param string: the text line corresponding to a Node
    :return: dictionnary of the different attributes of a Node (leaf of split)
    :rtype: dict
    """

    # pattern to look for in the strings
    patterns = {'variable': '\[(\w*)<',
                'threshold': '<(\d*.?\d*?)\]',
                'left_child': 'yes=(\d*),',
                'right_child': ',no=(\d*),',
                'missing': ',missing=(\d*),',
                'gain': ',gain=(\d*.?\d*?),',
                'cover': ',cover=(\d*.?\d*?)',
                'number': '^(\d*):',
                'leaf_score': 'leaf=(-?\d*.?\d*?)'
                }

    # result computation
    result = dict()

    for key, pattern in patterns.items():
        if re.search(pattern, string) is not None:
            if key in ['gain', 'cover', 'leaf_score', 'threshold']:
                result[key] = float(re.compile(pattern).findall(string)[0])
            else:
                result[key] = re.compile(pattern).findall(string)[0]

    return result


class NotSupportedModelError(Exception):
    def __init__(self,*args,**kwargs):
        Exception.__init__(self,*args,**kwargs)
