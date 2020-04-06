from random import shuffle
from copy import copy
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F

# List of values is converted into a binary tree for O(log V) training step complexity
# Nodes represent softmax layer points and not individual values in a tree

class Tree:
    def __init__(self, values):
        outputs = copy(values) # deepcopy
        shuffle(outputs)

        while len(outputs) > 2:
            temp_outputs = []
            for i in range(0, len(outputs), 2):
                if len(outputs) - (i+1) > 0:
                    temp_outputs.append([outputs[i], outputs[i+1]])
                else:
                    temp_outputs.append(outputs[i])
            outputs = temp_outputs

        self.tree = outputs
        self._count_nodes_dict = {}
                
    def _get_subtrees(self, tree):
        yield tree
        
        for subtree in tree:
            if type(subtree) == list:
                for x in self._get_subtrees(subtree):
                    yield x

    def _get_leaves_paths(self, tree):
        # binary tree so i can be only 0 or 1
        for i, subtree in enumerate(tree):

            if type(subtree) == list:
                for path, value in self._get_leaves_paths(subtree):
                    yield [i] + path, value
            else:
                # add terminal values finally
                yield [i], subtree
    
    
    def _count_nodes(self, tree):
        
        # id works as dictionary in Python is implemented as a hash map.
        # this step implements memoization
        if id(tree) in self._count_nodes_dict:
            return self._count_nodes_dict[id(tree)]
        
        size = 0
        for node in tree:
            if type(node) == list:
                size += 1 + self._count_nodes(node)

        return size

    # Returns all the nodes in a path
    def _get_nodes(self, tree, path):
        next_node = 0
        nodes = []
        
        for decision in path:
            nodes.append(next_node)

            # in case you go right node number would be all nodes skipped in left hand side
            next_node += 1 + self._count_nodes(tree[:decision])

            # step to left or right subtree
            tree = tree[decision]
        
        return nodes

class hier_softmax(nn.Module):
    def __init__(self, tree, context_dim=4):
        super(hier_softmax, self).__init__()
        self.tree = tree
        self.node2layer = {}

        # create a softmax linear layer for each node in the tree
        for i, subtree in enumerate(self.tree._get_subtrees(tree.tree)):
            self.node2layer["softmax_node_"+str(i)] = nn.Linear(context_dim, len(subtree))
        
        # create a dictionary mapping each terminal value to its path (turn directions and nodes)
        # only needs to be built once for all guture inferences

        value_to_path_and_nodes_dict = {}

        for path, value in self.tree._get_leaves_paths(tree.tree):
            nodes = self.tree._get_nodes(tree.tree, path)
            value_to_path_and_nodes_dict[chr(value)] = path, nodes
        
        self.value_to_path_and_nodes_dict = value_to_path_and_nodes_dict
        
        self.tree = tree
    
    def forward(self, context, value):
        probs = 1
        path, nodes = self.value_to_path_and_nodes_dict[chr(value)]
        for p, n in zip(path, nodes):
            inp = self.node2layer["softmax_node_"+str(n)](context)
            probs *= F.softmax(inp, dim=-1)[p]
        
        # print only probability of requested value. Value location found through path
        return probs


if __name__=='__main__':

    softree = Tree(list(range(10)))
    print(softree.tree)

    hsoft = hier_softmax(softree)
    print(hsoft(torch.randn(4), 7))

