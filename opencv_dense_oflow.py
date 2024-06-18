# Lucas-Kanade method computes optical flow for a sparse feature set 
# (in our example, corners detected using Shi-Tomasi algorithm). OpenCV 
# provides another algorithm to find the dense optical flow. It computes the
# optical flow for all the points in the frame. It is based on Gunnar
# Farneback's algorithm which is explained in "Two-Frame Motion Estimation
# Based on Polynomial Expansion" by Gunnar Farneback in 2003.
#
# Below sample shows how to find the dense optical flow using above algorithm.
# We get a 2-channel array with optical flow vectors, (u,v). We find their
# magnitude and direction. We color code the result for better visualization.
# Direction corresponds to Hue value of the image. Magnitude corresponds to
# Value plane. See the code below.

import numpy as np
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
        The example file can be downloaded from: \
        https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args()

cap = cv.VideoCapture(args.image)
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while(1):
    ret, frame2 = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0) # For speed, consider cv.optflow.DISOpticalFlow
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imshow('frame2', bgr)
    k = cv.waitKey(30) & 0xff

    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png', frame2)
        cv.imwrite('opticalhsv.png', bgr)

    prvs = next

cv.destroyAllWindows()