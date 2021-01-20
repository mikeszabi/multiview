# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 13:49:46 2018

@author: SzMike
"""

import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

from importlib import reload

run_id='20180808_150659_calib'

base_dir=r'd:\DATA\EON1\caronboardcalibration'
#base_dir=r'C:\\Users\\fodrasz\\'

calib_dir=os.path.join(base_dir,run_id,'start_calib')

cal_path_left           = os.path.join(calib_dir,'left')
cal_path_right           = os.path.join(calib_dir,'right')

left_tag='/*_1'
right_tag='/*_2'
ext='jpg' #'ppm'

chessboard_shape=[5,9]

images_right = glob.glob(cal_path_right + right_tag+'.'+ext)
images_left = glob.glob(cal_path_left + left_tag+'.'+ext)
images_left.sort()
images_right.sort()

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros(((chessboard_shape[0]-1)*(chessboard_shape[1]-1),3), np.float32)
objp[:,:2] = np.mgrid[0:(chessboard_shape[1]-1),0:(chessboard_shape[0]-1)].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
#out_file='calibration_20190329'+'.avi'
#out = cv2.VideoWriter(out_file,fourcc, 1, ( 4096,2176),True)



cv2.namedWindow('calib',cv2.WINDOW_NORMAL)
cv2.resizeWindow('calib',272,512)

for fname in images_right:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#    (h, w) = img.shape[:2]
#    # calculate the center of the image
#    center = (w / 2, h / 2)    
#    M = cv2.getRotationMatrix2D(center, 90, 1.0)
#    rotated90 = cv2.warpAffine(img, M, (h, w))
# 

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, ((chessboard_shape[1]-1),(chessboard_shape[0]-1)),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, ((chessboard_shape[1]-1),(chessboard_shape[0]-1)), corners2,ret)
        cv2.imshow('calib',img)
        cv2.waitKey(500)
        

#        out.write(img)

cv2.destroyAllWindows()
#out.release()

# Calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

fovx, fovy, focalLength, principalPoint, aspectRatio = cv2.calibrationMatrixValues(mtx, gray.shape[::-1], 1.0, 1.0)
 

# Reprojection error
tot_error  = 0
for i, obj_pts in enumerate(objpoints):
    imgpoints2, _ = cv2.projectPoints(obj_pts, rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error
    print(error)

print('total error: '+str(tot_error/len(objpoints)))

#####
#https://gist.github.com/aarmea/629e59ac7b640a60340145809b1c9013
# Stereo calibration
import src_stereo_calib.stereo_calib
reload(src_stereo_calib.stereo_calib)
from src_stereo_calib.stereo_calib import StereoCalibration 

#
cal_data = StereoCalibration()
cal_data.read_images(images_left,images_right)

import numpy as np
np.save('camera_model.npy', cal_data.camera_model) 
