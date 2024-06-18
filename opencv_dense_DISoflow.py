# https://docs.opencv.org/4.10.0/de/d4f/classcv_1_1DISOpticalFlow.html#details
# DIS optical flow algorithm.

# This class implements the Dense Inverse Search (DIS) optical flow algorithm.
# More details about the algorithm can be found at [150] . Includes three
# presets with preselected parameters to provide reasonable trade-off between
# speed and quality. However, even the slowest preset is still relatively fast,
# use DeepFlow if you need better quality and don't care about speed.

# This implementation includes several additional features compared to the
# algorithm described in the paper, including spatial propagation of flow
# vectors (getUseSpatialPropagation), as well as an option to utilize an 
# initial flow approximation passed to calc (which is, essentially, temporal
# propagation, if the previous frame's flow field is passed).


import numpy as np
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='This sample demonstrates DIS Optical Flow calculation. \
        The example file can be downloaded from: \
        https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args()

cap = cv.VideoCapture(args.image)
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

# print(dir(cv.DISOpticalFlow))
flowwer = 	cv.DISOpticalFlow.create(0) # PRESET_ULTRAFAST = 0 , PRESET_FAST = 1 , PRESET_MEDIUM = 2

while(1):
    ret, frame2 = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = flowwer.calc(prvs,next,None) #cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0) # For speed, consider cv.optflow.DISOpticalFlow
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