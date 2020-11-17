# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 10:17:50 2019

@author: szabo
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 15:32:52 2019

@author: mikeszabi
"""
import sys
import os
import cv2
import numpy as np

import file_helper as fh

base_dir=r'e:\DATA\BRAUN\kalib_cam1-4'
image_dir=os.path.join(base_dir,'cam4_jav')


# Leopard camera: 4056x3040 
# Matrixvision camera: 2176x4096
cam_size=(3692,5544)
th_scale=0.1
th_size=(int(th_scale*cam_size[0]),int(th_scale*cam_size[1])) # 2176x4096
f_mm=12 # focal length in mm
pixel_size_um=2.4 # pixel size
chessboard_shape=[7,10]

out_file=os.path.join(image_dir,'calibration.avi')
calib_mtx_file=os.path.join(image_dir,'calib_mtx')


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros(((chessboard_shape[0]-1)*(chessboard_shape[1]-1),3), np.float32)
objp[:,:2] = np.mgrid[0:(chessboard_shape[1]-1),0:(chessboard_shape[0]-1)].T.reshape(-1,2)

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

image_files=fh.imagelist_in_depth(image_dir,level=0,date_sort=False)

image_files=[image_file for image_file in image_files if os.path.basename(image_file).startswith('cam')]

############################
gridsize=10
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(gridsize,gridsize))
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


out = cv2.VideoWriter(out_file,fourcc, 1, th_size,True)

for i, image_file in enumerate(image_files):
    
    if not i%1==0:
        continue
    print(image_file)
    im=cv2.imread(image_file)

    im=cv2.resize(im, th_size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    
     # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, ((chessboard_shape[1]-1),(chessboard_shape[0]-1)),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        im = cv2.drawChessboardCorners(im, ((chessboard_shape[1]-1),(chessboard_shape[0]-1)), corners2,ret)
        
        out.write(im)
    else:
        print('Unsuccessful')
    cv2.imshow('calib',im)
    cv2.waitKey(10)

cv2.destroyAllWindows()
out.release()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

np.savez(calib_mtx_file, ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

fovx, fovy, focalLength, principalPoint, aspectRatio = cv2.calibrationMatrixValues(mtx, gray.shape[::-1], 1.0, 1.0)
 

 # Reprojection error
tot_error  = 0
for i, obj_pts in enumerate(objpoints):
    imgpoints2, _ = cv2.projectPoints(obj_pts, rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error
    print(error)

print('total error: '+str(tot_error/len(objpoints)))

print('number of successfull images: '+str(len(objpoints)))

#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

# undistort

# generator = block_blob_service.list_blobs(container_name, prefix=prefix,delimiter='/')
# for blob in generator:
#     print("\t Blob name: " + blob.name)

#     if not blob.name.split('.')[-1]=='jpg':
#         continue
    
#     blob = block_blob_service.get_blob_to_bytes(container_name,blob.name)
#     img = Image.open(BytesIO(blob.content))
#     img=img.resize(th_size)
    
#     im=np.asarray(img,dtype='uint8')
#     h,  w = im.shape[:2]
#     newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h)) 
#     # undistort
#     dst = cv2.undistort(im, mtx, dist, None, newcameramtx)
    
#     # crop the image
#     x,y,w,h = roi
#     dst = dst[y:y+h, x:x+w]
#     cv2.imshow('calib',im)
#     cv2.waitKey(-1)
    
#     # undistort
#     mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
#     dst = cv2.remap(im,mapx,mapy,cv2.INTER_LINEAR)
    
#     # crop the image
#     x,y,w,h = roi
#     dst = dst[y:y+h, x:x+w]
#     cv2.imshow('calib',dst)
#     cv2.waitKey(-1)