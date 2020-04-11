import random
import numpy as np 
from copy import deepcopy
from timeit import timeit
from time import time 

def naive_sampling(num_samples, sample_size, elements, probabilities):
    samples = []
    for _ in range(num_samples):
        sample = []
        candidate_elements = deepcopy(elements)        

        while len(sample) < sample_size:
            candidate_element = random.choice(candidate_elements)            

            # probs consists of inclusion probabilities
            if probabilities[candidate_element] >= random.random():
                sample.append(candidate_element)
                candidate_elements.remove(candidate_element)        

        samples.append(sample)
    return samples

def fast_sampling(num_samples, sample_size, elements, probabilities):

    prob_dict = probabilities
    # unpacking probabilites dictionary
    probabilities = list(prob_dict.values())

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
    num_samples = 1000
    sample_size = 5

    elements = list(np.random.randint(low=0,high=100,size=num_elements))
    probabilities = list(np.random.rand(num_elements))
    probabilities = dict(zip(elements,probabilities))

    # SETUP_CODE = ""

    # TEST_CODE = '''naive_sampling(num_samples, sample_size, elements, probabilities)
    # '''
    # print(timeit(setup=SETUP_CODE,stmt=TEST_CODE,number=1))

    start = time()
    samples = naive_sampling(num_samples, sample_size, elements, probabilities)
    print(time()-start)

    start = time()
    samples = fast_sampling(num_samples, sample_size, elements, probabilities)
    print(time()-start)

    # print(samples)