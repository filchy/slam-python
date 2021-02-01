from slam import process
from display import Display
from pointmap import PointMap

import cv2
import open3d as o3d

pmap = PointMap()
display = Display()

def main():
	cap = cv2.VideoCapture("videos/test_video3.mp4")

	pcd = o3d.geometry.PointCloud()
	visualizer = o3d.visualization.Visualizer()
	visualizer.create_window(window_name="3D plot", width=960, height=540)

	while cap.isOpened():
		ret, frame = cap.read()
		frame = cv2.resize(frame, (960, 540))
		img, tripoints, kpts, matches = process(frame)
		xyz = pmap.collect_points(tripoints)

		if ret:
			if kpts is not None or matches is not None:
				display.display_points2d(frame, kpts, matches)
			else:
				pass
			display.display_vid(frame)

			if xyz is not None:
				display.display_points3d(xyz, pcd, visualizer)
			else:
				pass
			if cv2.waitKey(1) == 27:
				break
		else:
			break

	cv2.destroyAllWindows()
	cap.release()

if __name__ == '__main__':
	main()
