import cv2
import numpy as np
vidcap = cv2.VideoCapture('G:\Misc\data\OFlow_CWpan_leftmotion.mp4')
success,image = vidcap.read()
count = 0
# while success:
# #   cv2.imwrite("G:\Misc\data\OFlow_CWpan_leftmotion_frame%d.jpg" % count, image)     # save frame as JPEG file      
#   success,image = vidcap.read()
#   print('Read a new frame: ', success)
#   count += 1