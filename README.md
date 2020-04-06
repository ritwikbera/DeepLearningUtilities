# Deep Learning Utilities
Low-level implementations of some interesting functions found in deep learning

### Implementations of theoretical concepts

* _Weighted Sampling with Segment Trees_: Tree based random weighted sampling for _O(log N)_ sampling time complexity. These can be updated on the fly after an initial build of the tree.
* _Negative Sampling_ : Generate fixed set of negative samples to be paired with equal number of positive samples for balanced training on unbalanced datasets.
* _Alias Method for Efficient Discrete Sampling_ : Constant time sampling method for discrete distributions.
* _Beam Search_ : Used a lot in language models' inference for most likely sequence decoding
* _K-Means Clustering based Layer-wise Weight Quantization_ : Idea introduced in _Deep Compression_ paper to reduce number of unique weights to be stored for a NN model.
* _Heirarchical Softmax_ : Generalizable softmax with _O(log V)_ training step complexity, owing to binary tree style construction. Inference still requires traversal of all possible tree paths to detect most likely output. Used in _word2vec__.
* _Node2Vec_: Generating random walks on graph networks to generate skipgram-style node embeddings. Note: (need to)
* _KD Tree_: A K dimensional tree data structure with function provided for nearest neighbor search. Used a lot in information retrieval/similarity search applications like __Spotify's *Annoy*__ and __Waymo's dataset *Content Search*__ tool.
* _Locally Sensitive Hashing_: LSH is used in deep learning in similarity search applications. It is a spatial hashing technique which ensures spatially close vectors are assigned the same hash value. Used in __Shazam__, __Uber's fraud detecton tool__, __Google's Reformer__ transformer architecture among others.
* _Knowledge Distillation_: Template for knowledge distillation training. Used in __Parallel WaveNet__ training, among other to reduce model size. Useful only for models with softmax outputs. Anneal temperature as training progresses for stable gradients.

### Debugging Tools

* _Memory Profiling with PyTorch Hooks_ : Useful in optimizing models, can help visualize memory usage during checkpointed training as well (Note: This is is not working with latest PyTorch update).
* _Unit test to verify parameter behavior during training_ : Useful while training GANs or during transfer learning where only a certain subset of parameters need to be tuned during training.