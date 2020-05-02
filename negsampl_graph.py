import networkx as nx 
import numpy as np 

'''
PyTorch BigGraph : Simple demo of batched negative sampling

Single Relation Graph but the same concept applies to Multi Relation Graphs where 
edges will be sampled from one dimension of the adjacency matrix at a time.

Facebook's implementation splits a sampled batch of edges into multiple chunks and forms
the i_th batch by taking the i_th element of each chunk and proceeds as below.
'''
class Node(object):
	def __init__(self, value):
		self.value = value

	def __mul__(self, other):
		try:
			assert type(other) == Node
		except AssertionError:
			other = Node(other)
		return Edge(Node(self.value), other)

	def __str__(self):
		return '{}'.format(self.value)

	__rmul__ = __mul__

class Edge(object):
	def __init__(self, node1, node2):
		self.node1 = node1
		self.node2 = node2

	def __eq__(self, other):
		if self.node1.value == other.node1.value and self.node2.value == other.node2.value:
			return True
		elif self.node2.value == other.node1.value and self.node1.value == other.node2.value:
			return True
		else:
			return False

	def __hash__(self):
		return id(str(self))

	def __str__(self):
		return '[{} {}]'.format(self.node1.value, self.node2.value)

G = nx.karate_club_graph()
edges = np.array(list(map(lambda edge: Edge(Node(edge[0]), Node(edge[1])), G.edges)))
batch = edges[np.random.choice(len(edges), 10, replace=True)]

edges_ = np.array(list(map(lambda edge: [Node(edge[0]), Node(edge[1])], G.edges)))
batch_ = edges_[np.random.choice(len(edges_), 10, replace=True)]

sources = batch_[:,0]
dests = batch_[:,1]
pairs = np.reshape(np.outer(sources, dests),-1)
no_edges = list(set(pairs).difference(edges))

[print(x) for x in no_edges]

assert G.has_edge(no_edges[0].node1.value, no_edges[0].node2.value) == False

