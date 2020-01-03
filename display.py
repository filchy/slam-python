import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *
from mpl_toolkits.mplot3d import Axes3D




class Display:
	def __init__(self):
		self.W = 960
		self.H = 540
		self.counts = 0
		self.last_x = []
		self.last_y = []
		self.last_z = []

	def display_points2d(self, img, kpts, matches):
		for kpt in kpts:
			cv2.circle(img, (int(kpt.pt[0]), int(kpt.pt[1])), radius=2, color=(0,255,0), thickness=-1)

		"""if matches != []:
									x1, y1 = matches[0][0], matches[0][1]
									x2, y2 = matches[1][0], matches[1][1]
						
									cv2.line(img, (x1, y1), (x2, y2), color=(255,0,0), thickness=1)"""
		
		for match in matches:
			(u1, v1) = np.int32(match[0].pt)
			(u2, v2) = np.int32(match[1].pt)
			cv2.line(img, (u1, v1), (u2, v2), color=(0,0,255), thickness=1)

		return img


	def display_points3d(self, tripoints3d):
		x_points = [pt for pt in tripoints3d[0]]
		y_points = [pt for pt in tripoints3d[1]]
		z_points = [pt for pt in tripoints3d[2]]

		if self.last_x != []:
			fig = plt.figure(figsize=(9,9))
			fig.suptitle('3D reconstructed', fontsize=16)
			ax = fig.gca(projection='3d')
			ax.plot(self.last_x, self.last_y, self.last_z, 'b.')
			ax.set_xlabel('x axis')
			ax.set_ylabel('y axis')
			ax.set_zlabel('z axis')
		
			plt.show()


		self.last_x = x_points
		self.last_y = y_points
		self.last_z = z_points
		self.counts += 1
		

	def display_vid(self, img):
		#if img.shape[:2] != (self.H, self.W):

			#img = cv2.resize(img, (960, 540))
		#print(img.shape)
		cv2.imshow("main", img)
