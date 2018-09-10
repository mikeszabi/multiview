# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import re

base_dir=r'd:\DATA\EON1\caronboardcalibration'
base_dir=r'd:\DATA\EON1\caronboardcalibration\3d_results\Cibakhaza'
run_id='20180808_153914'
run_id='art_20180823_121650_995'
ext='.jpg'

im_dir=os.path.join(base_dir,run_id)
jpg_list = [os.path.join(im_dir,f) for f in os.listdir(im_dir) if re.match('roi.*_.\.jpg', f)]


img1 = cv2.imread(jpg_list[5],0)  #queryimage # left image
img2 = cv2.imread(jpg_list[6],0) #trainimage # right image

#height, width = img1.shape[:2]
#img1 = cv2.resize(img1,(int(0.5*width), int(0.5*height)), interpolation = cv2.INTER_CUBIC)
#img2 = cv2.resize(img2,(int(0.5*width), int(0.5*height)), interpolation = cv2.INTER_CUBIC)
#img1.shape

## SIFT
#sift = cv2.xfeatures2d.SIFT_create(1000)
#
## find the keypoints and descriptors with SIFT
##gray1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
##gray2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
#
#
#kp1, des1 = sift.detectAndCompute(img1,None)
#kp2, des2 = sift.detectAndCompute(img2,None)
#
## FLANN parameters
#FLANN_INDEX_KDTREE = 0
#index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#search_params = dict(checks=50)
#
#flann = cv2.FlannBasedMatcher(index_params,search_params)
#matches = flann.knnMatch(des1,des2,k=1)
#matches = [m[0] for m in matches]


## ORB
orb = cv2.ORB_create(1000)

kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


matches = bf.match(des1,des2)


######
dmatches = sorted(matches, key = lambda x:x.distance)

## extract the matched keypoints
src_pts  = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
dst_pts  = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)


## find homography matrix and do perspective transform
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,1.0)
np.linalg.det(M)

h,w = img1.shape[:2]
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#pts = np.float32([ [282,634],[335,634],[335,664],[282,664] ]).reshape(-1,1,2)

dst = cv2.perspectiveTransform(pts,M)

## draw found regions

img2_poly = cv2.polylines(img2, [np.int32(dst)], True, (0,0,255), 10, cv2.LINE_AA)
plt.imshow(img2_poly)

plt.figure(3)

## draw match lines
res = cv2.drawMatches(img1, kp1, img2, kp2, dmatches[:100],None,flags=2)

plt.imshow(res,cmap='gray')


#
#good = []
#pts1 = []
#pts2 = []
#
## ratio test as per Lowe's paper
#for i,(m,n) in enumerate(matches):
#    if m.distance < 0.8*n.distance:
#        good.append(m)
#        pts2.append(kp2[m.trainIdx].pt)
#        pts1.append(kp1[m.queryIdx].pt)
        

plt.figure(4)
img1_2 = cv2.warpPerspective(img1, M, (img2.shape[1],img2.shape[0]))
f, axarr = plt.subplots(1, 2)
axarr[0].imshow(img2,cmap='gray')
axarr[1].imshow(img1_2,cmap='gray')


#img1_kp=cv2.drawKeypoints(img1,kp1,img1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
