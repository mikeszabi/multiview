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

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH    = 6

def init_feature(name,n_feature_point=None):
    if n_feature_point is None:
        sift_default=1000
        surf_default=1000
        orb_default=500
    else:
        sift_default=n_feature_point
        surf_default=n_feature_point
        orb_default=n_feature_point        
        
    chunks = name.split('-')
    if chunks[0] == 'sift':
        detector = cv2.xfeatures2d.SIFT_create(sift_default)
        norm = cv2.NORM_L2
    elif chunks[0] == 'surf':
        detector = cv2.xfeatures2d.SURF_create(surf_default)
        norm = cv2.NORM_L2
    elif chunks[0] == 'orb':
        detector = cv2.ORB_create(orb_default)
        norm = cv2.NORM_HAMMING
    elif chunks[0] == 'akaze':
        detector = cv2.AKAZE_create()
        norm = cv2.NORM_HAMMING
    elif chunks[0] == 'brisk':
        detector = cv2.BRISK_create()
        norm = cv2.NORM_HAMMING
    else:
        return None, None
    if 'flann' in chunks:
        if norm == cv2.NORM_L2:
            flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        else:
            flann_params= dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 6, # 12
                               key_size = 12,     # 20
                               multi_probe_level = 1) #2
        matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    else:
        matcher = cv2.BFMatcher(norm)
    return detector, matcher

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape[:2]
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1[0]),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2[0]),5,color,-1)
    return img1,img2

det_type='orb'
ratio=0.75


#base_dir=r'd:\Data\EON1\photosFor3D\SceauxCastle'
#base_dir=r'd:\DATA\EON1\caronboardcalibration\3d_results\Cibakhaza'
base_dir=r'd:\DATA\MAV1\Images'
run_id='art_20180823_121650_995'
#run_id='images'
run_id='Selection_20190124_120409_2800_3400'
ext='.jpg'

im_dir=os.path.join(base_dir,run_id)
#jpg_list = [os.path.join(im_dir,f) for f in os.listdir(im_dir) if re.match('roi.*_.\.jpg', f)]

jpg_list = imsz.imagelist_in_depth(im_dir,level=1)

i1=2
im_file1=jpg_list[i1]
i2=3 
im_file2=jpg_list[i2]
im_file1=r'd:\\DATA\\MAV1\\Images\\Selection_20190124_120409_2800_3400\\roi_0_2882.jpg'
im_file2=r'd:\\DATA\\MAV1\\Images\\Selection_20190124_120409_2800_3400\\roi_1_2882.jpg'

img1 = cv2.imread(im_file1,0)  #queryimage # left image
img2 = cv2.imread(im_file2,0) #trainimage # right image


##
max_dim=2028
img1, scale = imsz.imRescaleMaxDim(img1, max_dim, boUpscale = False, interpolation = 1)
img2, scale = imsz.imRescaleMaxDim(img2, max_dim, boUpscale = False, interpolation = 1)

## SIFT
# sift = cv2.xfeatures2d.SIFT_create(800)

detector,matcher=init_feature(det_type,n_feature_point=2000)

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
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
np.linalg.det(M)


################# transform points
h,w = img1.shape[:2]
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)
## draw found regions

#img2_poly = cv2.polylines(img2, [np.int32(dst)], False, (0,0,255), 10, cv2.LINE_AA)
############## plot result        
img1_2 = cv2.warpPerspective(img1, M, (img2.shape[1],img2.shape[0]))


fig=plt.figure(2)
axarr = fig.subplots(1, 3)
axarr[0].imshow(img1,cmap='gray')
axarr[1].imshow(img1_2,cmap='gray')
axarr[2].imshow(img2,cmap='gray')

#################
# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
pts1 = np.int32(src_pts)
pts2 = np.int32(dst_pts)
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_8POINT  ) # FM_LMEDS

# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]


## Find epilines corresponding to points in left image (first image) and
## drawing its lines on right image
#lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
#lines1 = lines1.reshape(-1,3)
#img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
#
## Find epilines corresponding to points in left image (first image) and
## drawing its lines on right image
#lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
#lines2 = lines2.reshape(-1,3)
#img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
#
#fig=plt.figure(3)
#axarr = fig.subplots(1, 2)
#axarr[0].imshow(img5,cmap='gray')
#axarr[1].imshow(img3,cmap='gray')
#plt.show()

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