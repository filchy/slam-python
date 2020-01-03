from slam import slam_run

import cv2


def main():
	cap = cv2.VideoCapture("test_video2.mp4")

	while cap.isOpened():
		ret, frame = cap.read()

		slam_run(frame)

		if cv2.waitKey(1) == 27:
			cv2.destroyAllWindows()
			cap.release()
			break

if __name__ == '__main__':
	main()
