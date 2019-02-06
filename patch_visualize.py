# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 11:16:52 2018

@author: szabo
"""

import cv2

import numpy as np
from imtools_msz import imRescaleMaxDim

#https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
refPt_transformed = []
cropping = False

i1=0
i2=1

im1=jpg_list[i1]
im2=jpg_list[i2]

M=hcobj.Ms[im1,im2]

img1 = cv2.imread(im1,0)  #queryimage # left image
img2 = cv2.imread(im2,0)  #queryimage # left image        

max_dim=hcobj.image_width
img1, scale = imRescaleMaxDim(img1, max_dim, boUpscale = False, interpolation = 1)
img2, scale = imRescaleMaxDim(img2, max_dim, boUpscale = False, interpolation = 1)

img2 = cv2.warpPerspective(img1, M, (img2.shape[1],img2.shape[0]))

# M[0][2]*=scale
# M[1][2]*=scale
# M[2][0]*=scale
# M[2][1]*=scale


def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
    global refPt, refPt_transformed, cropping, M, w
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
 
	# check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
        refPt.append((x, y))
        cropping = False
 
    # draw a rectangle around the region of interest
        cv2.rectangle(clone, refPt[0], refPt[1], (0, 255, 0), 2)
        
        refPt_transformed = cv2.perspectiveTransform(np.float32([refPt]), M)[0]
        refPt_transformed[0][0]+=w
        refPt_transformed[1][0]+=w
      
        cv2.rectangle(clone, tuple(refPt_transformed[0]), tuple(refPt_transformed[1]), (0, 255, 0), 2)

        cv2.imshow("image", clone)

# load the image, clone it, and setup the mouse callback function
        

clone=np.concatenate((img1, img2), axis=1)


#cv2.namedWindow("image")
cv2.namedWindow("image",cv2.WINDOW_NORMAL)
cv2.resizeWindow('image',(img1.shape[1],int(img1.shape[0]/2)))
cv2.setMouseCallback("image", click_and_crop)
 
# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", clone)
	key = cv2.waitKey(1) & 0xFF
 
	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		image = clone.copy()
 
	# if the 'c' key is pressed, break from the loop
	elif key == ord("c"):
		break
 
# if there are two reference points, then crop the region of interest
# from teh image and display it
#if len(refPt) == 2:
#    roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
#    print(roi.shape)
#    roi1_2 = clone[int(refPt[0][1]):int(refPt[1][1]), int(refPt[0][0]+w):int(refPt[1][0]+w)]
#    print(roi1_2.shape)
##    refPt = []
#    cv2.imshow("ROI", np.concatenate((roi, roi1_2), axis=1))
#    cv2.waitKey(0)
 
# close all open windows
cv2.destroyAllWindows()  