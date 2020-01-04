import numpy as np


def scale_and_transform_points(points):
	x = points[0]
	y = points[1]

	center = points.mean(axis=1)

	cx = x - center[0]
	cy = y - center[1]

	distance = np.sqrt(np.power(cx, 2) + np.power(cy, 2))
	scale = np.sqrt(2) / distance.mean()

	norm3d = np.array([
		[scale, 0, -scale*center[0]],	#x
		[0, scale, -scale*center[1]],	#y
		[0, 0, 1]])			#z

	return np.dot(norm3d, points), norm3d


def correspondence_matrix(p1, p2):
	p1x, p1y = p1[:2]
	p2x, p2y = p2[:2]

	return np.array([
		p1x * p2x, p1x * p2y, p1x,
		p1y * p2x, p1y * p2y, p1y,
		p2x, p2y, np.ones(len(p1x))
		]).T

	return np.array([
		p2x * p1x, p2x * p1y, p2x,
        p2y * p1x, p2y * p1y, p2y,
        p1x, p1y, np.ones(len(p1x))
        ]).T


def compute_img_to_img_matrix(x1, x2, compute_essential=False):
	A = correspondence_matrix(x1, x2)
	U, S, V = np.linalg.svd(A)
	F = V[-1].reshape(3, 3)

	U, S, V = np.linalg.svd(F)
	S[-1] = 0
	if compute_essential:
		S = [1, 1, 0] # Force rank 2 and equal eigenvalues
	F = np.dot(U, np.dot(np.diag(S), V))

	return F


def compute_essential_normalized_matrix(p1, p2, compute_essential=False):
	if p1.shape != p2.shape:
		raise ValueError("Numbers of p1 and p2 don´t match !")

	# preprocess img coords
	p1n, T1 = scale_and_transform_points(p1)
	p2n, T2 = scale_and_transform_points(p2)

	# compute F
	F = compute_img_to_img_matrix(p1n, p2n, compute_essential)

	F = np.dot(T1.T, np.dot(F, T2))

	F = F / F[2, 2]

	return F


def compute_essential_normalized(p1, p2):
	return compute_essential_normalized_matrix(p1, p2, compute_essential=True)


def compute_P_from_essential(E):
	U, S, V = np.linalg.svd(E)

	if np.linalg.det(np.dot(U, V)) < 0:
		V = -V

	W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
	P2s = [
		np.vstack((np.dot(U, np.dot(W, V)).T, U[:, 2])).T,
		np.vstack((np.dot(U, np.dot(W, V)).T, -U[:, 2])).T,
		np.vstack((np.dot(U, np.dot(W.T, V)).T, U[:, 2])).T,
		np.vstack((np.dot(U, np.dot(W.T, V)).T, -U[:, 2])).T]

	return P2s


def skew(x):
	return np.array([
		[0, -x[2], x[1]],
		[x[2], 0, -x[0]],
		[-x[1], x[0], 0]])


def reconstruct_one_point(pt1, pt2, m1, m2):
	A = np.vstack([
		np.dot(skew(pt1), m1),
		np.dot(skew(pt2), m2)])

	U, S, V = np.linalg.svd(A)
	P = np.ravel(V[-1, :4])

	return P / P[3]


def triangulation(p1, p2, m1, m2):
	num_points = p1.shape[1]
	res = np.ones((4, num_points))

	for i in range(num_points):
		A = np.asarray([
			(p1[0, i] * m1[2, :] - m1[0, :]),
			(p1[1, i] * m1[2, :] - m1[1, :]),
			(p2[0, i] * m2[2, :] - m2[0, :]),
			(p2[1, i] * m2[2, :] - m2[1, :])])

		_, _, V = np.linalg.svd(A)
		X = V[-1, :4]
		res[:, i] = X / X[3]

	return res
