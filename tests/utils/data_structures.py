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
    labels = ['label_0', 'label_1', 'label_2', 'label_3']
    root = example_tree.root
    assert example_tree.size == 7
    assert example_tree.labels == labels
    assert example_tree.get_labels([0,1,2,3]) == labels
    assert example_tree.count_leaves() == 4
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
    df = example_tree.cut(1, use_labels=True)
    assert all([i in ['label_0', 'label_1', 'label_2', 'label_3'] for i in df.index])
    assert len(df) == 4
    assert df.nunique() == 1
    assert df.iloc[0] == 6  # Root cluster

def test_cut_k2(example_tree):
    df = example_tree.cut(2, use_labels=True)
    assert all([i in ['label_0', 'label_1', 'label_2', 'label_3'] for i in df.index])
    assert len(df) == 4
    assert df.nunique() == 2
    
    cluster1 = df[df == 1]
    assert cluster1.iloc[0] == 1
    
    cluster5 = df[df.index != 'label_1']
    assert all(cluster5 == 5)

def test_cut_k3(example_tree):

    # use_labels=True
    df = example_tree.cut(3, use_labels=True)
    assert all([i in ['label_0', 'label_1', 'label_2', 'label_3'] for i in df.index])
    assert len(df) == 4
    assert df.nunique() == 3
    
    assert df[df.index ==  'label_0'].iloc[0] == 0
    assert df[df.index == 'label_1'].iloc[0] == 1
    cluster4 = df[df.index.isin(['label_2', 'label_3'])]
    assert all(cluster4 == 4)

    # use_labels=False
    df = example_tree.cut(3, use_labels=False)
    assert all([i in [0,1,2,3] for i in df.index])
    assert len(df) == 4
    assert df.nunique() == 3
    
    assert df[df.index == 0].iloc[0] == 0
    assert df[df.index == 1].iloc[0] == 1
    cluster4 = df[df.index.isin([2,3])]
    assert all(cluster4 == 4)

def test_cut_k4(example_tree):

    # use_labels=False
    labels = [0,1,2,3]
    df = example_tree.cut(4, use_labels=False)
    print(df)
    assert all([i in labels for i in df.index])
    assert len(df) == 4
    assert df.nunique() == 4

    for label in labels:
        cluster_id = df[df.index == label].iloc[0]
        assert label == cluster_id

    # use_labels=True
    labels = ['label_0', 'label_1', 'label_2', 'label_3']
    df = example_tree.cut(4, use_labels=True)
    assert all([i in labels for i in df.index])
    assert len(df) == 4
    assert df.nunique() == 4

    for label in labels:
        cluster_id = df[df.index == label].iloc[0]
        assert label == f'label_{cluster_id}'
