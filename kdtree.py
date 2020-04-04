import heapq
import itertools
import operator
import math
from collections import deque
import random

class Node(object):
    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right


    @property
    def is_leaf(self):
        return (not self.data) or (all(not bool(c) for c, p in self.children))

    def inorder(self):
        if not self:
            return

        if self.left:
            for x in self.left.inorder():
                yield x

        yield self

        if self.right:
            for x in self.right.inorder():
                yield x

    @property
    def children(self):
        if self.left and self.left.data is not None:
            yield self.left, 0
        if self.right and self.right.data is not None:
            yield self.right, 1

        if index == 0:
            self.left = child
        else:
            self.right = child

    def height(self):
        min_height = int(bool(self))
        return max([min_height] + [c.height()+1 for c, p in self.children])

    def __nonzero__(self):
        return self.data is not None

    __bool__ = __nonzero__

    def __eq__(self, other):
        if isinstance(other, tuple):
            return self.data == other
        else:
            return self.data == other.data

class KDNode(Node):
    def __init__(self, data=None, left=None, right=None, axis=None, dimensions=None):
        super(KDNode, self).__init__(data, left, right)
        self.axis = axis
        self.dimensions = dimensions

    def add(self, point):
        current = self # start from the top and traverse down
        while True:
            if self.data is None:
                current.data = point
                return current

            if point[current.axis] < current.data[current.axis]:
                if current.left is None:
                    current.left = current.create_subnode(point)
                    return current.left
                else:
                    current = current.left
            else:
                if current.right is None:
                    current.right = current.create_subnode(point)
                    return current.right
                else:
                    current = current.right

    def create_subnode(self, data):
        return self.__class__(data,
                axis=self.sel_axis,
                sel_axis=(self.sel_axis+1)%self.dimensions,
                dimensions=self.dimensions)

    def axis_dist(self, point, axis):
        return abs(self.data[axis] - point[axis])

    def dist(self, point):
        r = range(self.dimensions)
        return sum([self.axis_dist(point, i) for i in r])

    def _search_node(self, point, k, results, counter):
        if not self:
            return

        nodeDist = self.dist(point)

        # Add current node to the priority queue if it closer than
        # at least one point in the queue.
        #
        # If the heap is at its capacity, we need to check if the
        # current node is closer than the current farthest node, and if
        # so, replace it.
        item = (-nodeDist, next(counter), self)
        if len(results) >= k:
            if -nodeDist > results[0][0]:
                heapq.heapreplace(results, item)
        else:
            heapq.heappush(results, item)

        plane_dist = abs(point[self.axis] - self.data[self.axis])
  
        if point[self.axis] < self.data[self.axis]:
            if self.left is not None:
                self.left._search_node(point, k, results, counter)
        else:
            if self.right is not None:
                self.right._search_node(point, k, results, counter)

        # Search the other side of the splitting plane if it may contain
        # points closer than the farthest point in the current results.
        if -plane_dist > results[0][0] or len(results) < k:
            if point[self.axis] < self.data[self.axis]:
                if self.right is not None:
                    self.right._search_node(point, k, results, counter)
            else:
                if self.left is not None:
                    self.left._search_node(point, k, results, counter)

def create(point_list, dimensions, axis=0):

    if len(point_list)==0:
        return None

    point_list = list(point_list)
    point_list.sort(key=lambda point: point[axis])
    median = len(point_list) // 2

    root_data   = point_list[median]
    left  = create(point_list[:median], dimensions, axis=(axis+1)%dimensions)
    right = create(point_list[median + 1:], dimensions, axis=(axis+1)%dimensions)
    
    return KDNode(root_data, left, right, axis=axis, dimensions=dimensions)

if __name__=='__main__':
    results = []
    num_points = 10
    dimensions = 2
    k = 5
    point_list = []

    for i in range(num_points):
        point_list.append([random.randint(0,100), random.randint(0,100)])

    root = create(point_list, dimensions)
    search_point = [30,40]
    root._search_node(search_point, k, results, itertools.count())

    sorted_results = [(node.data, -d) for d, _, node in sorted(results, reverse=True)]
    print(sorted_results)