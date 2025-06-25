import pandas as pd
import numpy as np

class BinaryClusterTree():

    class Node():

        def __init__(self, value=None, height=None):
            self.value = value
            self.height = height
            self.left = None
            self.right = None

        def __str__(self):
            left_val = self.left.value if self.left else None
            right_val = self.right.value if self.right else None
            repr = f'Node {self.value}: children: left={left_val}, right={right_val}'
            return repr

        def is_leaf(self):
            return self.left is None and self.right is None

    def __init__(self, constructor=None):

        self.root = None
        self.size = 0
        self.labels = None
        self.constructor = constructor

        if constructor:
            self.labels = constructor['labels']
            self._init_from_constructor(constructor)

    def __len__(self):
        return self.size

    def find(self, value):
        node =  BinaryClusterTree._find(self.root, value)

        if node is None:
            raise KeyError(value)

        return node.value

    def __getitem__(self, value):
        return self.find(value)

    def get_leaves(self):
        return BinaryClusterTree._get_leaves(self.root)

    def count_leaves(self):
        return BinaryClusterTree._count_leaves(self.root)

    def cut(self, k, use_labels=False):
        
        if self.root is None:
            raise ValueError()

        cluster_leaves_map = BinaryClusterTree._cut(self.root, k)

        cluster_pep_map = {}
        for cluster_id, leaf_nodes in cluster_leaves_map.items():
            cluster_pep_map[cluster_id] = [n.value for n in leaf_nodes]

        pep_cluster_map = {p: c for c, peps in cluster_pep_map.items() for p in peps}

        ds = pd.Series(pep_cluster_map)

        if use_labels:
            cluster_label_map = {i: self.labels[i] for i in ds.index}
            ds.index = ds.index.map(cluster_label_map)

        return ds

    @staticmethod
    def _cut(node, k):

        if k <= 0:
            raise ValueError()

        queue = [node]

        for i in range(1, k):

            candidates = []  # list of (index_in_queue, node)
            for idx, n in enumerate(queue):
                if not n.is_leaf():
                    candidates.append((idx, n))

            if not candidates:
                break

            max_height = -1
            candidate_idx_in_queue = None
            for (idx, n) in candidates:
                if n.height > max_height:
                    max_height = n.height
                    candidate_idx_in_queue = idx

            node_to_split = queue.pop(candidate_idx_in_queue)
            queue.append(node_to_split.left)
            queue.append(node_to_split.right)

        cluster_leaves_map = {}
        for cluster_node in queue:
            leaves = BinaryClusterTree._get_leaves(cluster_node)
            cluster_leaves_map[cluster_node.value] = leaves

        return cluster_leaves_map

    def get_labels(self, cluster_ids):
        n_leaves = BinaryClusterTree._count_leaves(self.root)

        if np.any(np.array(cluster_ids) > n_leaves - 1):
            raise ValueError()

        return [self.labels[i] for i in cluster_ids]

    def _init_from_constructor(self, constructor):

         match constructor['type']:

            case  'sklearn_agglomerative_clustering':
                # Create binary tree from sklearn.cluster.AgglomerativeClustering object
                # From leaves up to root
                # leaves -> labels
                # nodes -> cluster numbers

                children = constructor['merge']

                if not children:
                    raise ValueError(constructor['merge'])

                #labels = constructor['labels']
                heights = constructor['heights']
                n_samples = len(children) + 1 #len(labels) # == len(merge) + 1

                # The root is the last merge operation
                self.root = self._build_sklearn_tree(children, heights, n_samples, len(children) - 1)
                self.size += n_samples + len(children)

            case _:
                raise ValueError('Constructor type not supported')

    def print_tree(self):

        if self.root is None:
            print("Empty tree")
            return
    
        BinaryClusterTree._print_node(self.root, labels=self.labels)

    @staticmethod
    def _find(node, value):
        '''Breadth first approach'''
        if node is None:
            return None

        queue = [node]

        while queue:

            current = queue.pop(0)

            if current.value == value:
                return current

            if current.left is not None:
                queue.append(current.left)

            if current.right is not None:
                queue.append(current.right)

        return None

    @staticmethod
    def _get_leaves(node):

        if node is None:
            return []

        leaves = []
        queue = [node]

        while queue:

            current = queue.pop(0)

            if current.is_leaf():
                leaves.append(current)

            else:
                if current.left is not None:
                    queue.append(current.left)
                if current.right is not None:
                    queue.append(current.right)

        return leaves

    @staticmethod
    def _count_leaves(node):
        return len(BinaryClusterTree._get_leaves(node))

    @staticmethod
    def count_children(node):
        if node is None:
            raise ValueError(node)

        count = (node.left is not None) + (node.right is not None)

        if count not in (0,2):
            raise ValueError('There are not 0 or 2 children')

        return count

    def _build_sklearn_tree(self, children, heights, n_samples, merge_idx):
        if merge_idx < 0:
            raise ValueError(merge_idx)
            
        # Current merge creates node with value = n_samples + merge_idx (cluster ID)
        left_child_id, right_child_id = children[merge_idx]
        node = BinaryClusterTree.Node(value=n_samples + merge_idx,
                                      height=heights[merge_idx]) # value= n_samples + merge_idx,
        #value=n_samples - merge_idx - 2,

        
        # Handle left child
        if left_child_id < n_samples:
            # Left child is a leaf (original sample)
            node.left = BinaryClusterTree.Node(value=left_child_id,
                                               height=heights[merge_idx])
            
        else:
            # Left child is an internal node, recurse
            child_merge_idx = left_child_id - n_samples
            node.left = self._build_sklearn_tree(children, heights, n_samples, child_merge_idx)
            
        # Handle right child  
        if right_child_id < n_samples:
            # Right child is a leaf (original sample)
            node.right = BinaryClusterTree.Node(value=right_child_id,
                                                height=heights[merge_idx])

        else:
            # Right child is an internal node, recurse
            child_merge_idx = right_child_id - n_samples
            node.right = self._build_sklearn_tree(children, heights, n_samples, child_merge_idx)
            
        return node

    @staticmethod
    def _print_node(node, indent=0, labels=None):

        if node is None:
            return
        
        # Print current node with indentation
        height = node.height if node.height else ''
        label = ' "' + str(labels[node.value]) + '"' if labels and node.is_leaf() else ''
        print('(' + str(height) + ')' + '    ' * indent + str(node.value) + label)
        
        # Print children with increased indentation
        if node.left is not None or node.right is not None:

            if node.left is not None:
                BinaryClusterTree._print_node(node.left, indent + 1, labels=labels)

            else:
                print('  ' * (indent + 1) + 'None')
                
            if node.right is not None:
                BinaryClusterTree._print_node(node.right, indent + 1, labels=labels)

            else:
                print('  ' * (indent + 1) + 'None')
