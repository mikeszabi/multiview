# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 14:52:51 2018

@author: SzMike
"""

import cv2



def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

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

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])


cv2.namedWindow('render',cv2.WINDOW_NORMAL)
cv2.resizeWindow('render',272,512)    
    
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out_file='render_cube_20190329'+'.avi'
out = cv2.VideoWriter(out_file,fourcc, 1, ( 1088,2048),True)



for fname in images_right:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, ((chessboard_shape[1]-1),(chessboard_shape[0]-1)),None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Find the rotation and translation vectors.
        ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        img = draw(img,corners2,imgpts)
        cv2.imshow('render',img)
        k = cv2.waitKey(0) & 0xff
        out.write(img)

#        if k == 's':
#            print('frame added')
##            cv2.imwrite(fname[:6]+'.png', img)
#            out.write(img)

out.release()
cv2.destroyAllWindows()
