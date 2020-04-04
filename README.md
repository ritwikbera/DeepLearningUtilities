# Deep Learning Utilities
Low-level implementations of some interesting functions found in deep learning

### Implementations of theoretical concepts

* _Negative Sampling_ : Generate fixed set of negative samples to be paired with equal number of positive samples for balanced training on unbalanced datasets.
* _Alias Method for Efficient Discrete Sampling_ : Constant time sampling method for discrete distributions.
* _Beam Search_ : Used a lot in language models' inference for most likely sequence decoding
* _K-Means Clustering based Layer-wise Weight Quantization_ : Idea introduced in _Deep Compression_ paper to reduce number of unique weights to be stored for a NN model.
* _Node2Vec_: Generating random walks on graph networks to generate skipgram-style node embeddings. Note: (need to)
* _KD Tree_: A K dimensional tree data structure with function provided for nearest neighbor search. Used a lot in information retrieval applications like __Spotify's *Annoy*__ and __Waymo's dataset *Content Search*__ tool.

### Debugging Tools

* _Memory Profiling with PyTorch Hooks_ : Useful in optimizing models, can help visualize memory usage during checkpointed training as well (Note: This is is not working with latest PyTorch update).
* _Unit test to verify parameter behavior during training_ : Useful while training GANs or during transfer learning where only a certain subset of parameters need to be tuned during training.