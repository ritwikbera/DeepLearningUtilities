import numpy as np 
import heapq
import random

def reservoir_sampling(stream, size):
    i, sample = 0, []

    for item in stream:
        
        i += 1
        k = int(random.random()*i)

        if len(sample) < size:
            sample.append(item)
        
        elif k < size:
            sample[k] = item

    return sample

def weight(item):
    return 5 if item%2==0 else 1

def weighted_reservoir_sampling(stream, size):

    heap = [] 

    for item in stream:
        wi = weight(item) # item[1]
        ui = random.uniform(0, 1)
        ki = ui ** (1/wi)

        if len(heap) < size:
            heapq.heappush(heap, (ki, item))
        elif ki > heap[0][0]:
            heapq.heappush(heap, (ki, item))

            if len(heap) > size:
                heapq.heappop(heap)

    return [sample[1] for sample in heap]

if __name__=='__main__':
    stream = [i for i in range(1000)]
    size = 5
    reservoir = reservoir_sampling(stream, size)
    print(reservoir)
    reservoir = weighted_reservoir_sampling(stream, size)
    print(reservoir)
