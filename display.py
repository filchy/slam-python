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
		
		if matches !=0:
			for match in matches:
				(u1, v1) = np.int32(match[0].pt)
				(u2, v2) = np.int32(match[1].pt)
				cv2.line(img, (u1, v1), (u2, v2), color=(0,0,255), thickness=1)

		return img


	def display_points3d(self, tripoints3d):
		"""
		# for matplotlib
		x_points = [pt for pt in tripoints3d[0]]
		y_points = [-pt for pt in tripoints3d[1]]
		z_points = [-pt for pt in tripoints3d[2]]

		if self.last_x != []:
			fig = plt.figure(figsize=(9,9))
			fig.suptitle('3D reconstructed', fontsize=16)
			ax = fig.gca(projection='3d')
			ax.plot(self.last_x, self.last_z, self.last_y, 'b.')
			ax.set_xlabel('x axis')
			ax.set_ylabel('z axis')
			ax.set_zlabel('y axis')
		
			plt.show()

		self.last_x = x_points
		self.last_y = y_points
		self.last_z = z_points
		self.counts += 1
		"""
		if tripoints3d is not None:
			pcd = o3d.geometry.PointCloud()
			pcd.points = o3d.utility.Vector3dVector(tripoints3d)
			o3d.visualization.draw_geometries([pcd])

	def display_vid(self, img):

		cv2.imshow("main", img)
