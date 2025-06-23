import pytest
from copro.utils.data_structures import BinaryClusterTree

@pytest.fixture
def example_tree():
    constructor = {
        'type': 'sklearn_agglomerative_clustering',
        'labels': ['label_0', 'label_1', 'label_2', 'label_3'],
        'merge': [[2, 3], [4, 0], [1, 5]],
        'heights': [0.1, 0.3, 0.8]
    }
    return BinaryClusterTree(constructor)

def test_tree_structure(example_tree):
    root = example_tree.root
    assert root.value == 6
    assert root.height == 0.8
    
    assert root.left.value == 1
    assert root.left.height == 0.8
    assert root.left.is_leaf()
    
    assert root.right.value == 5
    assert root.right.height == 0.3
    assert not root.right.is_leaf()
    
    node5 = root.right
    assert node5.left.value == 4
    assert node5.left.height == 0.1
    assert not node5.left.is_leaf()
    
    assert node5.right.value == 0
    assert node5.right.height == 0.3
    assert node5.right.is_leaf()
    
    node4 = node5.left
    assert node4.left.value == 2
    assert node4.left.height == 0.1
    assert node4.left.is_leaf()
    
    assert node4.right.value == 3
    assert node4.right.height == 0.1
    assert node4.right.is_leaf()

def test_count_leaves(example_tree):
    assert example_tree.count_leaves() == 4
    
    root = example_tree.root
    node5 = root.right
    assert BinaryClusterTree._count_leaves(node5) == 3  # Leaves: 0,2,3
    
    node4 = node5.left
    assert BinaryClusterTree._count_leaves(node4) == 2  # Leaves: 2,3

def test_cut_k1(example_tree):
    df = example_tree.cut(1)
    assert len(df) == 4
    assert df['cluster_id'].nunique() == 1
    assert df['cluster_id'].iloc[0] == 6  # Root cluster

def test_cut_k2(example_tree):
    df = example_tree.cut(2)
    assert len(df) == 4
    assert df['cluster_id'].nunique() == 2
    
    cluster1 = df[df['sample'] == 1]
    assert cluster1['cluster_id'].iloc[0] == 1
    
    cluster5 = df[df['sample'] != 1]
    assert all(cluster5['cluster_id'] == 5)

def test_cut_k3(example_tree):
    df = example_tree.cut(3)
    assert len(df) == 4
    assert df['cluster_id'].nunique() == 3
    
    assert df[df['sample'] == 1]['cluster_id'].iloc[0] == 1
    assert df[df['sample'] == 0]['cluster_id'].iloc[0] == 0
    cluster4 = df[df['sample'].isin([2,3])]
    assert all(cluster4['cluster_id'] == 4)

def test_cut_k4(example_tree):
    df = example_tree.cut(4)
    assert len(df) == 4
    assert df['cluster_id'].nunique() == 4

    for sample in range(4):
        cluster_id = df[df['sample'] == sample]['cluster_id'].iloc[0]
        assert cluster_id == sample
