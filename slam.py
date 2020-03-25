from display import Display
from extractor import Extractor
from convertor import cart2hom
from normalize import compute_essential_normalized, compute_P_from_essential, reconstruct_one_point, triangulation


import cv2
import numpy as np
import open3d as o3d


display = Display()
extractor = Extractor()


def process(img):
	pts1, pts2, kpts, matches = extractor.extract_keypoints(img=img)

	# converto to 3 dimensional
	points1 = cart2hom(pts1)
	points2 = cart2hom(pts2)

	img_h, img_w, img_ch = img.shape

	intrinsic = np.array([[3000,0,img_w/2],
				[0,3000,img_h/2],
				[0,0,1]])
	tripoints3d = []
	if points1.ndim != 1 or points2.ndim != 1:
		points1_norm = np.dot(np.linalg.inv(intrinsic), points1)
		points2_norm = np.dot(np.linalg.inv(intrinsic), points2)

		E = compute_essential_normalized(points1_norm, points2_norm)

		P1 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
		P2s = compute_P_from_essential(E)

		ind = -1
		for i, P2 in enumerate(P2s):
			d1 = reconstruct_one_point(points1_norm[:, 0], points2_norm[:, 0], P1, P2)

			P2_homogenous = np.linalg.inv(np.vstack([P2, [0,0,0,1]]))

			d2 = np.dot(P2_homogenous[:3, :4], d1)

			if d1[2] > 0 and d2[2] > 0:
				ind = i

		P2 = np.linalg.inv(np.vstack([P2s[ind], [0, 0, 0, 1]]))[:3, :4]
		tripoints3d = triangulation(points1_norm, points2_norm, P1, P2)

	else:
		print("Wrong dimension of array")
		pass

	return img, tripoints3d, kpts, matches
