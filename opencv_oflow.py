# OpenCV provides all these in a single function, cv.calcOpticalFlowPyrLK().
# Here, we create a simple application which tracks some points in a video.
# To decide the points, we use cv.goodFeaturesToTrack(). We take the first
# frame, detect some Shi-Tomasi corner points in it, then we iteratively
# track those points using Lucas-Kanade optical flow. For the function
# cv.calcOpticalFlowPyrLK() we pass the previous frame, previous points and
# next frame. It returns next points along with some status numbers which has
# a value of 1 if next point is found, else zero. We iteratively pass these
# next points as previous points in next step. See the code below.
#
# (This code doesn't check how correct are the next keypoints. So even if
# any feature point disappears in image, there is a chance that optical flow
# findsthe next point which may look close to it. So actually for a robust
# tracking, corner points should be detected in particular intervals. OpenCV
# samples comes up with such a sample which finds the feature points at every
# 5 frames. It also run a backward-check of the optical flow points got to 
# select only good ones. Check samples/python/lk_track.py).

# Note: read about image pyramids: https://docs.opencv.org/3.4/d4/d1f/tutorial_pyramids.html


import numpy as np
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
        The example file can be downloaded from: \
        https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args()
cap = cv.VideoCapture(args.image)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
        qualityLevel = 0.3,
        minDistance = 7,
        blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize = (15, 15),
        maxLevel = 2,
        criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
while(1):
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        img = cv.add(frame, mask)
        cv.imshow('frame', img)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
cv.destroyAllWindows()