import cv2
import numpy as np


def cart2hom(array):
	"""Convert array from Cartesian -> Homogenous (2 dimensions -> 3 dimensions)"""
	if array.ndim == 1:
		return np.array([0])

	else:
		array_3dim = np.asarray(np.vstack([array, np.ones(array.shape[1])]))
		return array_3dim
	
