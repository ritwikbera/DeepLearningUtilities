import random
import numpy as np

'''
SegmentTree stores priorities and data in different arrays and
has methods to update priorities
'''

class SegmentTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, threshold):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if threshold <= self.tree[left]:
            return self._retrieve(left, threshold)
        else:
            return self._retrieve(right, threshold - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, priority)

        self.write = (self.write+1)%self.capacity

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, threshold):
        idx = self._retrieve(0, threshold)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

class Memory:
    def __init__(self, capacity):
        self.tree = SegmentTree(capacity)
        self.capacity = capacity

    def add(self, priority, sample):
        self.tree.add(priority, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segsize = self.tree.total() / n
        priorities = []

# Generate upto n random samples from different sections of the data spread

        for i in range(n):
            (idx, p, data) = self.tree.get(random.uniform(segsize*i, segsize*(i+1)))
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        return batch, idxs

    def update(self, idx, priority):
        self.tree.update(idx, priority)

if __name__=='__main__':
    memory = Memory(10)
    memory.add(3,(1,1,1,1))
    memory.add(2,(0,4,1,1))
    memory.add(10,(1,3,4,2))

    batch, idxs = memory.sample(3)
    print(batch)
    print(idxs)

    memory.update(idxs[0], 30)
    
    batch, idxs = memory.sample(3)
    print(batch)
    print(idxs)