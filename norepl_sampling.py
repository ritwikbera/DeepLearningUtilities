import random
import numpy as np 
from copy import deepcopy
from time import time 
from sumtree_sampling import SegmentTree

def segtree_swor(num_samples, sample_size, elements, probabilities):
    samples = []
    tree = SegmentTree(len(elements))
    for i in range(len(elements)):
        tree.add(probabilities[i], elements[i])

    cached_probs = []

    for _ in range(num_samples):
        sample = []
        if len(cached_probs) > 0:
            [tree.update(idx, p) for (idx, p) in cached_probs]

        for _ in range(sample_size):
            (idx, p, data) = tree.get(random.uniform(0, tree.total()))
            cached_probs.append((idx, p))
            sample.append(data)
            tree.update(idx, 0)

        samples.append(sample)

    return samples

def naive_sampling(num_samples, sample_size, elements, probabilities):
    samples = []
    for _ in range(num_samples):
        sample = []
        candidate_elements = deepcopy(elements)        

        while len(sample) < sample_size:
            candidate_idx = random.choice(range(len(candidate_elements)))
            candidate_element = candidate_elements[candidate_idx]

            # probs consists of inclusion probabilities
            if probabilities[candidate_idx] >= random.random():
                sample.append(candidate_element)
                candidate_elements.remove(candidate_element)        

        samples.append(sample)
    return samples

def fast_sampling(num_samples, sample_size, elements, probabilities):

    elements = np.array(elements)

    # replicate probabilities as many times as `num_samples`
    replicated_probabilities = np.tile(probabilities, (num_samples, 1))    

    # get random shifting numbers & scale them correctly
    random_shifts = np.random.random(replicated_probabilities.shape)
    random_shifts /= random_shifts.sum(axis=1)[:, np.newaxis]    

    # shift by numbers & find largest (by finding the smallest of the negative)
    shifted_probabilities = random_shifts - replicated_probabilities

    # only ensure element at sample_size position is at correct sorted place
    # all elements to the left of it will be smaller by default then.
    return [elements[np.argpartition(shifted_probabilities[i], sample_size, axis=0)[:sample_size]] for i in range(num_samples)]

if __name__=='__main__':
    
    num_elements = 20
    num_samples = 5
    sample_size = 5

    elements = list(np.random.randint(low=0,high=100,size=num_elements))
    probabilities = list(np.random.rand(num_elements))

    start = time()
    samples = naive_sampling(num_samples, sample_size, elements, probabilities)
    print(time()-start)

    start = time()
    samples = fast_sampling(num_samples, sample_size, elements, probabilities)
    print(time()-start)

    samples = globals()['segtree_swor'](num_samples, sample_size, elements, probabilities)

    print(samples)