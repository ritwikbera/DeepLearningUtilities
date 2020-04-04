import numpy as np
from statistics import mode

# number of buckets equals 2 raised to number of bits in hash size

class HashTable:
    def __init__(self, hash_size, inp_dimensions):
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.hash_table = dict()
        self.projections = np.random.randn(self.hash_size, inp_dimensions)
        
    def generate_hash(self, inp_vector):
        bools = (np.dot(inp_vector, self.projections.T) > 0).astype('int')
        return ''.join(bools.astype('str'))

    def __setitem__(self, inp_vec, label):
        hash_value = self.generate_hash(inp_vec)
        self.hash_table[hash_value] = self.hash_table.get(hash_value, list()) + [label]
        
    def __getitem__(self, inp_vec):
        hash_value = self.generate_hash(inp_vec)
        return self.hash_table.get(hash_value, [])
 
# Ensemble of hash tables ensures better stability of similar item search
# It provides robustness to poorly selected random hyperplanes

class LSH:
    def __init__(self, num_tables, hash_size, inp_dimensions):
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.hash_tables = list()
        for i in range(self.num_tables):
            self.hash_tables.append(HashTable(self.hash_size, self.inp_dimensions))
    
    def __setitem__(self, inp_vec, label):
        for table in self.hash_tables:
            table.__setitem__(inp_vec, label)
    
    def __getitem__(self, inp_vec):
        results = list()
        for table in self.hash_tables:
            results.extend(table.__getitem__(inp_vec))
        return list(set(results))

if __name__=='__main__':
    inp_dimensions= 10
    hash_size = 4
    num_obs = 20
    num_tables = 1

    num_bins = 2**hash_size
    inputs = np.random.randn(num_obs, inp_dimensions)
    labels = np.random.randint(low=1, high=num_bins, size=num_obs)

    print(labels)

    lsh = LSH(num_tables, hash_size, inp_dimensions)

    for i in range(num_obs):
        lsh.__setitem__(inputs[i], labels[i])

    # test

    test_indices = np.random.randint(low=0, high=num_obs-1, size=5)

    print(test_indices)
    for i in test_indices:
        print(lsh.__getitem__(inputs[i]))



