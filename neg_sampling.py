import numpy as np 

def negsamp_vectorized_bsearch(pos_inds, n_items, n_samp=32):
	"""
	Negative Sampling via pre-verification

	pos_inds : list with indices of positive samples
	n_items : total number of items
	n_samp :  number of negative samples to be output
	"""

	raw_samp = np.random.randint(0,n_items-len(pos_inds),size=n_samp)
	pos_inds_adj = pos_inds - np.arange(len(pos_inds))
	ss = np.searchsorted(pos_inds_adj, raw_samp, side='right')
	neg_inds = raw_samp + ss

	return neg_inds

if __name__=='__main__':
	n_items = 10
	pos_inds = [3, 7]
	print(negsamp_vectorized_bsearch(pos_inds, n_items))

