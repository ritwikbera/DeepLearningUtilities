import numpy as np

# fast vectorized implementation
def countSketchInMemory(matrixA, s):
    m, n = matrixA.shape
    matrixC = np.zeros([m, s])
    hashedIndices = np.random.choice(s, n, replace=True)
    randSigns = np.random.choice(2, n, replace=True) * 2 - 1 # a n-by-1{+1, -1} vector
    matrixA = matrixA * randSigns.reshape(1, n) # flip the signs of 50% columns of A
    for i in range(s):
        idx = (hashedIndices == i)
        matrixC[:, i] = np.sum(matrixA[:, idx], 1)
    return matrixC

if __name__=='__main__':
	inputs = np.random.randint(0, high=100,size=(5,1600))
	print(inputs)

	# transform five 1600-dim vectors to 10 dim count sketches
	sketch_dim = 10
	sketches = countSketchInMemory(inputs, sketch_dim)
	print(sketches)

	# transformed sketch vectors should preserve geometric relativeness of original vectors
	# let's check the vector norms (original and transformed sketches)

	origNorms = np.sqrt(np.sum(np.square(inputs), 1))
	print(origNorms)
	sketchNorms = np.sqrt(np.sum(np.square(sketches), 1))
	print(sketchNorms)