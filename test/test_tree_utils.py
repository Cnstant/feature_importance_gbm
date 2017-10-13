import pytest
import pickle

from feature_importance_gbm.tree_utils import tree_cleaning, node_information


@pytest.fixture()
def example_tree():
    return pickle.load(open("./Notebooks/data/model.pickle.dat", "rb")).get_dump(with_stats=True)


def test_tree_cleaning_returns_the_right_list_of_strings(example_tree):
    string = tree_cleaning(example_tree[0])
    assert string[13] == '4:[var38<117563] yes=9,no=10,missing=9,gain=38.286,cover=1519'


def test_node_information_returns_a_leaf(example_tree):
    string = tree_cleaning(example_tree[45])
    assert node_information(string[6]) == {'cover': 5.0, 'leaf_score': -0.0, 'number': '53'}


def test_node_information_returns_a_node(example_tree) :
    string = tree_cleaning(example_tree[16])
    assert node_information(string[16]) == {'cover': 74.0,
                                            'gain': 4.2928,
                                            'left_child': '17',
                                            'missing': '17',
                                            'number': '8',
                                            'right_child': '18',
                                            'threshold': 65821.0,
                                            'variable': 'ID'}
