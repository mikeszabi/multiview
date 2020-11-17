# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 19:25:30 2020

@author: szabo
"""

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import file_helper as fh


#axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_pose/py_pose.html
#def draw(img, corners, imgpts):
#    corner = tuple(corners[0].ravel())
#    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
#    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
#    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
#    return img

def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img

base_dir=r'e:\DATA\BRAUN\kalib_cam1-4'
image_dir=os.path.join(base_dir,'cam1_jav')
calib_mtx_file=os.path.join(image_dir,'calib_mtx.npz')


# Leopard camera: 4056x3040 
# Matrixvision camera: 2176x4096
cam_size=(3692,5544)
th_scale=0.1
th_size=(int(th_scale*cam_size[0]),int(th_scale*cam_size[1])) # 2176x4096

chessboard_shape=[7,10]

f_mm=12 # focal length in mm
pixel_size_um=2.4 # pixel size
    
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros(((chessboard_shape[0]-1)*(chessboard_shape[1]-1),3), np.float32)
objp[:,:2] = np.mgrid[0:(chessboard_shape[1]-1),0:(chessboard_shape[0]-1)].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.



with np.load(calib_mtx_file) as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out_file='calibration'+'.avi'
out = cv2.VideoWriter(out_file,fourcc, 1, th_size,True)


image_files=fh.imagelist_in_depth(image_dir,level=1,date_sort=False)
image_files=[image_file for image_file in image_files if os.path.basename(image_file).startswith('cam')]


gridsize=10

clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(gridsize,gridsize))

i=0
for i, image_file in enumerate(image_files):
    
    if not i%1==0:
        continue
    print(image_file)
    im=cv2.imread(image_file)

    im=cv2.resize(im, th_size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    
     # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, ((chessboard_shape[1]-1),(chessboard_shape[0]-1)), flags=cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors.
        ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        im = draw(im,corners2,imgpts)
        cv2.imshow('render',im)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        
        out.write(im)

    else:
        print('Unsuccessful')
    cv2.imshow('calib',im)
    cv2.waitKey(10)

cv2.destroyAllWindows()
out.release()