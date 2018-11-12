# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 21:25:28 2018

@author: szabo
"""
from importlib import reload
import os
import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt
import src_stereo_calib.stereo_calib
reload(src_stereo_calib.stereo_calib)
from src_stereo_calib.stereo_calib import StereoCalibration 

#https://gist.github.com/aarmea/629e59ac7b640a60340145809b1c9013

run_id='20180808_161610_calib'

base_dir=r'd:\DATA\EON1\caronboardcalibration'
#base_dir=r'C:\\Users\\fodrasz\\'
im_dir=os.path.join(base_dir,run_id)

calib_dir=os.path.join(base_dir,run_id,'')

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

# Camera calibrations, stereo camera calibrations
cal_data = StereoCalibration(chessboard_shape=chessboard_shape)
cal_data.read_images(images_left,images_right)

np.save('camera_model.npy', cal_data.camera_model) 


camera_model = np.load('camera_model.npy').item()
OPTIMIZE_ALPHA = 0.25
(leftRectification, rightRectification, leftProjection, rightProjection,
        dispartityToDepthMap, leftROI, rightROI) = cv2.stereoRectify(
                camera_model['M1'], camera_model['dist1'],
                camera_model['M2'], camera_model['dist2'],
                camera_model['img_shape'], camera_model['R'], camera_model['T'],
                None, None, None, None, None,
                cv2.CALIB_ZERO_DISPARITY, OPTIMIZE_ALPHA)

leftMapX, leftMapY = cv2.initUndistortRectifyMap(
        camera_model['M1'], camera_model['dist1'], leftRectification,
        leftProjection, camera_model['img_shape'], cv2.CV_32FC1)
rightMapX, rightMapY = cv2.initUndistortRectifyMap(
        camera_model['M2'], camera_model['dist2'], rightRectification,
        rightProjection, camera_model['img_shape'], cv2.CV_32FC1)


"""
"""
run_id='20180808_150659_calib'

base_dir=r'd:\DATA\EON1\caronboardcalibration'
#base_dir=r'C:\\Users\\fodrasz\\'
im_dir=os.path.join(base_dir,run_id)
imgL = cv2.imread(os.path.join(im_dir,'roi1127_1.jpg'))
imgR = cv2.imread(os.path.join(im_dir,'roi1127_2.jpg'))


fixedLeft = cv2.remap(imgL, leftMapX, leftMapY, cv2.INTER_LINEAR)
fixedRight = cv2.remap(imgR, rightMapX, rightMapY, cv2.INTER_LINEAR)

stereoMatcher = cv2.StereoBM_create()
stereoMatcher.setMinDisparity(16)
stereoMatcher.setNumDisparities(32)
stereoMatcher.setBlockSize(15)

stereoMatcher.setSpeckleRange(16)
stereoMatcher.setSpeckleWindowSize(50)

grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
depth = stereoMatcher.compute(grayLeft, grayRight)

##

window_size = 5
min_disp = 0
num_disp = 96-min_disp
stereo = cv2.StereoSGBM_create(
    minDisparity = min_disp,
    numDisparities = num_disp,
    uniquenessRatio = 2,
    speckleWindowSize = 100,
    speckleRange = 32,
    disp12MaxDiff = 1,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2,
)

disparity = stereo.compute(grayLeft, grayRight).astype(np.float32) / 16.0
disparity = (disparity-min_disp)/num_disp

## disparity without rectification

grayLeft = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayRight = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
depth = stereoMatcher.compute(grayLeft, grayRight)

cv2.namedWindow('depth',cv2.WINDOW_NORMAL)
cv2.resizeWindow('depth',272,512)    

DEPTH_VISUALIZATION_SCALE = 1024
cv2.imshow('depth', depth / DEPTH_VISUALIZATION_SCALE)
cv2.waitKey(500) 