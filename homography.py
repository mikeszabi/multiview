# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import imtools_msz as imsz
import homography_chain as hc



det_type='orb'
ratio=0.75
max_dim=1014


#base_dir=r'd:\Data\EON1\photosFor3D\SceauxCastle'
base_dir=r'd:\DATA\EON\Tests\bbox_from_annot_2'
# base_dir=r'd:\DATA'
# run_id='MetalCom'
run_id='image'
#run_id='Selection_20190124_120409_2800_3400'
# ext='.jpg'

im_dir=os.path.join(base_dir,run_id)
#jpg_list = [os.path.join(im_dir,f) for f in os.listdir(im_dir) if re.match('roi.*_.\.jpg', f)]

jpg_list = imsz.imagelist_in_depth(im_dir,level=1)

i1=1
im_file1=jpg_list[i1]
i2=2
im_file2=jpg_list[i2]
#im_file1=r'd:\\DATA\\MAV1\\Images\\Selection_20190124_120409_2800_3400\\roi_0_2882.jpg'
#im_file2=r'd:\\DATA\\MAV1\\Images\\Selection_20190124_120409_2800_3400\\roi_1_2882.jpg'

img1 = cv2.imread(im_file1,0)  #queryimage # left image
img2 = cv2.imread(im_file2,0) #trainimage # right image


##
img1, scale = imsz.imRescaleMaxDim(img1, max_dim, boUpscale = False, interpolation = 1)
img2, scale = imsz.imRescaleMaxDim(img2, max_dim, boUpscale = False, interpolation = 1)

## SIFT
# sift = cv2.xfeatures2d.SIFT_create(800)

detector,matcher=hc.init_feature(det_type,n_feature_point=1000)

kp1, des1 = detector.detectAndCompute(img1,None)
kp2, des2 = detector.detectAndCompute(img2,None)

# draw rich keypoints
img1_kp=cv2.drawKeypoints(img1,kp1,img1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_kp=cv2.drawKeypoints(img2,kp2,img2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

fig=plt.figure(0)
axarr = fig.subplots(1, 2)
axarr[0].imshow(img1_kp,cmap='gray')
axarr[1].imshow(img2_kp,cmap='gray')

matches = matcher.knnMatch(des1,des2, k=2)


# ratio test as per Lowe's paper
# Apply ratio test
good = []

for m in matches:
    if len(m)<2:
        print(m)
    else:
        if m[0].distance < ratio*m[1].distance:
            good.append(m[0])

######
dmatches = sorted(good, key = lambda x:x.distance)


## draw match lines
res = cv2.drawMatches(img1, kp1, img2, kp2, dmatches[:1000],None,flags=2)
plt.figure(1)
plt.imshow(res,cmap='gray')


################# extract the matched and filtered keypoints
src_pts  = np.float32([kp1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
dst_pts  = np.float32([kp2[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)


################# find homography matrix and do perspective transform
if len(dmatches)>10:

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    np.linalg.det(M)
    
    
    # ################# transform points
    # h,w = img1.shape[:2]
    # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    # dst = cv2.perspectiveTransform(pts,M)
    # ## draw found regions
    
    # #img2_poly = cv2.polylines(img2, [np.int32(dst)], False, (0,0,255), 10, cv2.LINE_AA)
    # ############## plot result        
    # img1_2 = cv2.warpPerspective(img1, M, (img2.shape[1],img2.shape[0]))
    
    
    # fig=plt.figure(2)
    # axarr = fig.subplots(1, 3)
    # axarr[0].imshow(img1,cmap='gray')
    # axarr[1].imshow(img1_2,cmap='gray')
    # axarr[2].imshow(img2,cmap='gray')
    
    #################
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    src_pts=src_pts[0:25,:,:]
    dst_pts=dst_pts[0:25,:,:]
    
    pts1 = np.int32(src_pts)
    pts2 = np.int32(dst_pts)
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS  ) # FM_LMEDS
    
    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]
    
    
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = hc.drawlines(img1,img2,lines1,pts1,pts2)
    
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = hc.drawlines(img2,img1,lines2,pts2,pts1)
    
    fig=plt.figure(3)
    axarr = fig.subplots(1, 2)
    axarr[0].imshow(img5,cmap='gray')
    axarr[1].imshow(img3,cmap='gray')
    plt.show()
    
    # rectification
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    p1fNew = pts1.reshape((pts1.shape[0] * 2, 1))
    p2fNew = pts2.reshape((pts2.shape[0] * 2, 1))
    
    
    retBool ,rectmat1, rectmat2 = cv2.stereoRectifyUncalibrated(p1fNew,p2fNew,F,img1.shape[::-1])
    dst11 = cv2.warpPerspective(img1,rectmat1,img1.shape[::-1])
    dst22 = cv2.warpPerspective(img2,rectmat2,img1.shape[::-1])
    
    fig=plt.figure(4)
    axarr = fig.subplots(1, 2)
    axarr[0].imshow(dst11,cmap='gray')
    axarr[1].imshow(dst22,cmap='gray')
    plt.show()
    
    #calculation of the disparity
    stereoMatcher = cv2.StereoBM_create()
    stereoMatcher.setMinDisparity(0)
    stereoMatcher.setNumDisparities(128)
    stereoMatcher.setBlockSize(5)
    
    stereoMatcher.setSpeckleRange(16)
    stereoMatcher.setSpeckleWindowSize(100)
    
    
    disp =  stereoMatcher.compute(dst22.astype('uint8'), dst11.astype('uint8')).astype(np.float32)
    disp=disp-disp.min()
    disp=disp/disp.max()
    
    fig=plt.figure(5)
    plt.imshow(disp);plt.colorbar();plt.clim(disp.min(),disp.max()/4)#;plt.show()