# Deep Learning Concepts
Low-level implementations of some interesting concepts found in deep learning research.

## Algorithms

### Sampling Algorithms

* _Weighted Sampling with Segment Trees_: Tree based random weighted sampling for _O(log N)_ sampling time complexity. These can be updated on the fly after an initial build of the tree.
Useful while implementing Prioritized Experience Replay for RL applications. 
Segment trees require _O(log N)_ time as opposed to _O(1)_ time for the alias method, while sampling discrete distributions but the alias method cannot adapt to evolving distributions like segment trees can.

* _Negative Sampling_ : Generate fixed set of negative samples to be paired with equal number of positive samples for balanced training on unbalanced datasets.

* _Alias Method for Efficient Discrete Sampling_ : Constant time sampling method for discrete distributions by converting the multinomial sampling process into a binomial sampling process.
Requires linear (_O(N)_) time to initially build the alias setup.

* _Fast Random Sampling without Replacement_: Implements fast batched random sampling without replacement. No setup time needed. Only access time complexity exists which is _O(B log N)_ where B is batch size and N is number of elements to sample from. This technique may be used in training neural networks via boosting.
A second version with Segment Trees is also implemented which also has _O(log N)_ complexity.

* _Reservoir Sampling_: A randomized algorithm to sample from streaming data, where each incoming data point has an equal probability (mathematically provable) of being sampled. Both weighted and unweighted versions of reservoir sampling are provided. Weighted version could be useful when we want to sample elements according to occurence frequency (frequency could be found out through _Count-Min Sketch_) or some custom weighting scheme.

* _Shuffle_: Implements a randomized algorithm, the Fisher-Yates shuffle, to shuufle a dataset so that each data-point can be placed at each position with uniform probability. Perfectly random shuffling is necessary in deep learning, so that the model does not learn any artificial temporal characteristics across batches.

* _Node2Vec_: Generating random walks on graph networks to generate skipgram-style node embeddings. Node2Vec's randomized nature offers more efficient exploration of a graph than BFS/DFS. A single random walk of length l generates (l-k) length skipgrams for k nodes. Hence it efficiently employs sample reusability. Note: (need to add test code).

* _Negative Edge Sampling on Graph_ : Efficient negative sampling for edges in a graph by computing outer-product like combination of nodes in batch of edges and deleting the true edges from the computed batch of edges. This was introduced in __Facebook's *PyTorch BigGraph*__ embedding learning algorithm for large knowledge graphs.

* _Markov Chain Monte Carlo_: A Monte Carlo randomized algorithm, Metropoliton-Hastings, to sample from a distribution.

### Miscellaneous

* _Beam Search_ : A deterministic algorithm used a lot in language models' inference for most likely sequence decoding.

## Data Structures

#### Deterministic

* _KD Tree_: A K dimensional tree data structure with function provided for nearest neighbor search. Used a lot in information retrieval/similarity search applications like __Spotify's *Annoy*__ and __Waymo's dataset *Content Search*__ tool.

    * __Spotify__ uses a KD Forest with the splitting planes being random hyperplanes and not just the elementary axes. More on this can be found on [Erik Bernhardsson's blog](https://erikbern.com/2015/10/01/nearest-neighbors-and-vector-models-part-2-how-to-search-in-high-dimensional-spaces.html)

* _Heirarchical Softmax Tree_ : Generalizable softmax with _O(log V)_ training step complexity, owing to binary tree style construction. Inference still requires traversal of all possible tree paths to detect most likely output. This is used in training _word2vec_ models.

#### Probabilistic

* _Locally Sensitive Hashing_: LSH is used in deep learning in similarity search applications. It is a spatial hashing technique which ensures spatially close vectors are assigned the same hash value. Used in __Shazam__, __Uber's fraud detecton tool__, __Google's Reformer__ transformer architecture among others.

* _Tensor Sketch_: Count sketch-based hashed transformations of high dimensional vectors. They transform high dimensional vectors to a low dimensional space (via random hashing). The vectors in the transformed space have similar geometric norms and relative spatial separation, which makes them useful to be used in a kernel. [_Compact Bilinear Pooling_](https://arxiv.org/abs/1511.06062) uses this to effeciently calculate outer product of high dimensional inputs and thus model second-order interaction effects in multimodal deep learning.





