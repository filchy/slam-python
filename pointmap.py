from slam import process
from display import Display

import numpy as np
import open3d as o3d
import cv2


class PointMap(object):
	def __init__(self):
		self.array = [0,0,0]

	def collect_points(self, tripoints):
		if len(tripoints) > 0:
			array_to_project = np.array([0,0,0])

			x_points = [pt for pt in tripoints[0]]
			y_points = [-pt for pt in tripoints[1]]
			z_points = [-pt for pt in tripoints[2]]

			for i in range(tripoints.shape[1]):
				curr_array = np.array([x_points[i], y_points[i], z_points[i]])
				array_to_project = np.vstack((array_to_project, curr_array))

			array_to_project = array_to_project[1:, :]

			return array_to_project
        
