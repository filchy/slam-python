import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.pylab import *
from mpl_toolkits.mplot3d import Axes3D



class Display:
	def __init__(self):
		self.W = 960
		self.H = 540
		#self.counts = 0
		#self.last_x = []
		#self.last_y = []
		#self.last_z = []

	def display_points2d(self, img, kpts, matches):
		if kpts != 0:
			for kpt in kpts:
				cv2.circle(img, (int(kpt.pt[0]), int(kpt.pt[1])), radius=2, color=(0,255,0), thickness=-1)
		
		if matches != 0:
			for match in matches:
				(u1, v1) = np.int32(match[0].pt)
				(u2, v2) = np.int32(match[1].pt)
				cv2.line(img, (u1, v1), (u2, v2), color=(0,0,255), thickness=1)
		return img


	def display_points3d(self, tripoints3d, pcd, visualizer):
		# open3d
		if tripoints3d is not None:
			pcd.clear()
			pcd.points = o3d.utility.Vector3dVector(tripoints3d)
			visualizer.remove_geometry(pcd)
			visualizer.add_geometry(pcd)
			visualizer.poll_events()
			visualizer.update_renderer()
			time.sleep(.2)

	def display_vid(self, img):
		cv2.imshow("main", img)
