from slam import process
from display import Display
from pointmap import PointMap

import cv2


pmap = PointMap()
display = Display()

def main():
    cap = cv2.VideoCapture("test_video3.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (960, 540))
        img, tripoints, kpts, matches = process(frame)
        xyz = pmap.collect_points(tripoints)

        display.display_points2d(frame, kpts, matches)
        display.display_points3d(xyz)
        display.display_vid(frame)

        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            cap.release()
            break

if __name__ == '__main__':
    main()
