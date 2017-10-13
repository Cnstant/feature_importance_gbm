import pytest
import pickle


from feature_importance_gbm.tree_information import Node, Tree
from feature_importance_gbm.tree_utils import tree_cleaning, node_information


@pytest.fixture()
def example_tree():
    return pickle.load(open("./Notebooks/data/model.pickle.dat", "rb")).get_dump(with_stats=True)


def test_tree_instanciation(example_tree):
    tree = Tree(example_tree[4])
    assert tree.node_table['18'].variable == 'var3'

def test_node_instanciation(example_tree):
    string = tree_cleaning(example_tree[33])[20]
    node = Node(node_information(string))
    assert node.number == '64'


# def test_gradientboostingpaths_instanciation(model):
    # un test pour le sklearn
    # un test pour le xgboost
    # un test pour un modèle rien à voir de sklearn (type logit)

