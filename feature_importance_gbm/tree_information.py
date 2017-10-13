from feature_importance_gbm.tree_utils import tree_cleaning, node_information
from feature_importance_gbm.tree_utils import NotSupportedModelError

import xgboost
import pandas as pd


class Node(object):
    """
    Create an abstraction of a node for a binary tree
    :param raw_string of information
    :return: a binary Node
    :rtype: Node as dict
    """
    def __init__(self, *initial_data, **kwargs):
        self.right_child = None
        self.left_child = None
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])


class Tree(object):
    """
    Create several abstractions for a booster : a dictionnary of all the nodes, a dictionnary of all the paths possible
    :param tree_text: tree as raw text
    :return: node table, leaf table, booster number
    :rtype: dict, dict, int
    """
    def __init__(self, tree_text):
        tree_cleaned = tree_cleaning(tree_text)
        info_extracted = [node_information(string) for string in tree_cleaned]
        self.node_table = {node.number: node for node in
                           [Node(x) for x in info_extracted]}
        self.table_leaf = {path[-1]: path[:-1] for path in
                           [x.split('->') for x in self._binarytreepaths(self.node_table['0'])]}

    def _binarytreepaths(self, root):
        """
        Compute recursively all the paths from the root of a tree to each leaf
        :param root: the first node of a booster iteration
        :return: tree as a list of nodes
        :rtype: list of str
        """
        if root is None:
            return []
        if root.left_child is None and root.right_child is None:
            return [str(root.number)]
        # if left/right is None we'll get empty list anyway
        return [str(root.number) + '->' + l for l in
                self._binarytreepaths(self.node_table[root.right_child]) + self._binarytreepaths(
                    self.node_table[root.left_child])]

    def _gain_computation(self, iteration_number):
        """
        Generate a dataframe of features with associate gain in the tree
        :param iteration_number: the number of the iteration
        :return: a dataframe feature/gain
        :rtype: pandas.Dataframe
        """
        df = pd.DataFrame([(node.variable, node.gain) for _,node in self.node_table.items()
                           if node.left_child is not None], columns=['feature', 'gain'])
        return df.groupby('feature').sum().apply(lambda x: x/(iteration_number+1)**2).reset_index()

    def _variable_use(self):
        """
        Computes the number of times each variable is used in the tree
        :param None
        :return: a dataframe feature/number of use
        :rtype: pandas.Dataframe
        """
        list_feature = [node.variable for _,node in self.node_table.items() if node.left_child is not None]
        return pd.Series(list_feature, name='use').value_counts().reset_index()


class GradientBoostingPaths(object):
    """
    Create the forest of Trees (see also Tree) as a dictionnary of trees. Key is booster number, value is the Tree
    :param model: the gradient boosting model you want to investigate on
    :return: a collection of Trees
    :rtype: dict of Trees
    """
    def __init__(self, model):
        if type(model) not in [xgboost.core.Booster, xgboost.sklearn.XGBClassifier]:
            raise NotSupportedModelError("Only binary classification models from xgboost implementation are currently"
                                         " supported")
        self.model = model
        try:
            model_dump = model.get_dump(with_stats=True)
            self.forest = {iteration: Tree(booster) for iteration, booster in enumerate(model_dump)}
        except AttributeError:
            model_dump = model._Booster.get_dump(with_stats=True)
            self.forest = {iteration: Tree(booster) for iteration, booster in enumerate(model_dump)}

    def feature_importance(self):
        """
        Computes the feature importance for each variable according to the gain it provides at each booster iteration
        divided by the number of iteration squared
        :param None
        :return: a dataframe of scores for each variable of the dataset
        :rtype: pandas.Dataframe
        """
        df = pd.concat([tree._gain_computation(iteration) for iteration, tree in self.forest.items()])
        return df.groupby('feature').sum().reset_index()

    def feature_utilisation(self):
        """
        Computes the feature utilisation for each variable as the rate of use
        :param None
        :return: a dataframe of rates for each variable of the dataset
        :rtype: pandas.Dataframe
        """
        df = pd.concat([tree._variable_use() for _, tree in self.forest.items()])
        df_grouped = (df.groupby('index').sum()/float(df.groupby('index').sum().sum())).reset_index()
        return df_grouped.rename(columns={'index': 'feature', 'use': 'use_rate'})

    ### point to point feature importance





