import argparse
import cv2 as cv
import numpy as np
import time

parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
        The example file can be downloaded from: \
        https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args()
cap = cv.VideoCapture(args.image)

cap = cv.VideoCapture(args.image)
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

# print(dir(cv.DISOpticalFlow))
flowwer = 	cv.DISOpticalFlow.create(0) # PRESET_ULTRAFAST = 0 , PRESET_FAST = 1 , PRESET_MEDIUM = 2

# Create a mask image for drawing purposes
mask_base = np.zeros_like(frame1)

while(1):
    ret, frame2 = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = flowwer.calc(prvs,next,None)

    top_section_x = np.average(flow[0:360,...,0])
    top_section_y = np.average(flow[0:360,...,1])
    bot_section_x = np.average(flow[720:1080,...,0])
    bot_section_y = np.average(flow[720:1080,...,1])

    bckgnd_x = (top_section_x + bot_section_x)/2.0
    bckgnd_y = (top_section_y + bot_section_y)/2.0
    mid_section1_x = np.average(flow[360:720, 0:640, 0]) - bckgnd_x
    mid_section1_y = np.average(flow[360:720, 0:640, 1]) - bckgnd_y
    mid_section2_x = np.average(flow[360:720, 640:1280, 0]) - bckgnd_x
    mid_section2_y = np.average(flow[360:720, 640:1280, 1]) - bckgnd_y
    mid_section3_x = np.average(flow[360:720, 1280:1920, 0]) - bckgnd_x
    mid_section3_y = np.average(flow[360:720, 1280:1920, 1]) - bckgnd_y
    
    # print("x: ", top_section_x, " y: ", top_section_y)
    # mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    # hsv[..., 0] = ang*180/np.pi/2
    # hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    # bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    # cv.imshow('frame2', bgr)

    # draw the tracks
    top_section_scale = 1.0; #1.0/np.sqrt(top_section_x**2.0 + top_section_y**2.0)
    top_section_x_int = int(top_section_x*top_section_scale)+960
    top_section_y_int = int(top_section_y*top_section_scale)+180
    mask = cv.line(mask_base.copy(), (960, 180), (top_section_x_int, top_section_y_int), (127, 127, 127), 2)
    mid_section1_scale = 1.0; #1.0/np.sqrt(mid_section1_x**2.0 + mid_section1_y**2.0)
    mid_section1_x_int = int(mid_section1_x*mid_section1_scale)+320
    mid_section1_y_int = int(mid_section1_y*mid_section1_scale)+540
    mask = cv.line(mask, (320, 540), (mid_section1_x_int, mid_section1_y_int), (127, 127, 127), 2)
    mid_section2_scale = 1.0; #1.0/np.sqrt(mid_section2_x**2.0 + mid_section2_y**2.0)
    mid_section2_x_int = int(mid_section2_x*mid_section2_scale)+960
    mid_section2_y_int = int(mid_section2_y*mid_section2_scale)+540
    mask = cv.line(mask, (960, 540), (mid_section2_x_int, mid_section2_y_int), (127, 127, 127), 2)
    mid_section3_scale = 1.0; #1.0/np.sqrt(mid_section3_x**2.0 + mid_section3_y**2.0)
    mid_section3_x_int = int(mid_section3_x*mid_section3_scale)+1600
    mid_section3_y_int = int(mid_section3_y*mid_section3_scale)+540
    mask = cv.line(mask, (1600, 540), (mid_section3_x_int, mid_section3_y_int), (127, 127, 127), 2)
    bot_section_scale = 1.0; #1.0/np.sqrt(bot_section_x**2.0 + bot_section_y**2.0)
    bot_section_x_int = int(bot_section_x*bot_section_scale)+960
    bot_section_y_int = int(bot_section_y*bot_section_scale)+900
    mask = cv.line(mask, (960, 900), (bot_section_x_int, bot_section_y_int), (127, 127, 127), 2)
    
    
    img = cv.add(frame2, mask)
    cv.imshow('frame2', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    prvs = next

    time.sleep(.5)
    # break
cv.destroyAllWindows()