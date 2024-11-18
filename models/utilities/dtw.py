import numpy as np
from numba import njit
from tqdm import tqdm

@njit
def DTWdistance(x:np.ndarray, y:np.ndarray)->float:
	"""
	Computes dynamic time warping distance
	between two timeseries x, y. 
	A simple O(N^2) algorithm is used.

	See https://en.wikipedia.org/wiki/Dynamic_time_warping

	This allows to study lightcurves
	topologically in an instrumental-independent
	(and redshift-independent?) way

	Parameters
	----------
	x, y : list
		Input 1-dimensional timeseries

	Returns
	-------
	distance : int
		DTW distance
	"""

	n, m = x.size, y.size
	DTW = np.full((n, m), fill_value=np.inf, dtype=np.float64)
	DTW[0, 0] = 0

	for i in range(1,n):
		for j in range(1,m):
			cost = np.abs(x[i] - y[j])
			DTW[i, j] = cost + np.min(DTW[i-1:i+1, j-1:j+1])

	distance = DTW[n-1, m-1]
	#print(distance)
	return distance

def DTWdistanceMatrix(X:list)->np.ndarray:
	"""
	Computes DTW distance matrix for a given
	collection of timeseries

	Parameters
	----------
	X : list
		A list of timeseries

	Returns
	-------
	D : np.ndarray
		Distance matrix
	"""

	n_samples=len(X)
	D = np.zeros((n_samples, n_samples))

	for i in tqdm(range(n_samples)):
		for j in range(i):
			distance = DTWdistance(X[i], X[j])
			D[i, j] = D[j, i] = distance

	return D




