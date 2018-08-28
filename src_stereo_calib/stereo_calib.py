# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 13:54:20 2018

@author: SzMike
"""

import numpy as np
import cv2
import glob
import argparse

# https://github.com/bvnayak/stereo_calibration

class StereoCalibration(object):
    def __init__(self,chessboard_shape=[5,9]):
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS +
                             cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        self.chessboard_shape=chessboard_shape

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros(((self.chessboard_shape[0]-1)*(self.chessboard_shape[1]-1),3), np.float32)
        self.objp[:,:2] = np.mgrid[0:(self.chessboard_shape[1]-1),0:(self.chessboard_shape[0]-1)].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.
        
        self.img_shape=[]

    def read_images(self, images_left, images_right):

        cv2.namedWindow('calib',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('calib',272,512)    
        for i, fname in enumerate(images_right):
            print(i)
            img_l = cv2.imread(images_left[i])
            img_r = cv2.imread(images_right[i])

            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
            

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, ((self.chessboard_shape[1]-1), (self.chessboard_shape[0]-1)), None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, ((self.chessboard_shape[1]-1), (self.chessboard_shape[0]-1)), None)

            # If found, add object points, image points (after refining them)
            self.objpoints.append(self.objp)

            if ret_l is True:
                corners_l2 = cv2.cornerSubPix(gray_l, corners_l, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_l.append(corners_l2)

                # Draw and display the corners
                img_l = cv2.drawChessboardCorners(img_l, ((self.chessboard_shape[1]-1), (self.chessboard_shape[0]-1)),
                                                  corners_l, ret_l)

                cv2.imshow('calib', img_l)
                cv2.waitKey(500)

            if ret_r is True:
                corners_r2= cv2.cornerSubPix(gray_r, corners_r, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_r.append(corners_r2)

                # Draw and display the corners
                img_r = cv2.drawChessboardCorners(img_r, ((self.chessboard_shape[1]-1), (self.chessboard_shape[0]-1)),
                                                  corners_r2, ret_r)
                cv2.imshow('calib', img_r)
                cv2.waitKey(500)
            self.img_shape = gray_l.shape[::-1]

        # CALIBRATION
        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, self.img_shape, None, None)

        rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, self.img_shape, None, None)

        self.camera_model = self.stereo_calibrate(self.img_shape)

    def stereo_calibrate(self, dims):
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5

        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l,
            self.imgpoints_r, self.M1, self.d1, self.M2,
            self.d2, dims,
            criteria=stereocalib_criteria, flags=flags)

        print('Intrinsic_mtx_1', M1)
        print('dist_1', d1)
        print('Intrinsic_mtx_2', M2)
        print('dist_2', d2)
        print('R', R) # rotaton
        print('T', T) # translation
        print('E', E)
        print('F', F)

        # for i in range(len(self.r1)):
        #     print("--- pose[", i+1, "] ---")
        #     self.ext1, _ = cv2.Rodrigues(self.r1[i])
        #     self.ext2, _ = cv2.Rodrigues(self.r2[i])
        #     print('Ext1', self.ext1)
        #     print('Ext2', self.ext2)

        print('')

        camera_model = dict([('M1', M1), ('M2', M2), ('dist1', d1),
                            ('dist2', d2), ('rvecs1', self.r1),
                            ('rvecs2', self.r2), ('R', R), ('T', T),
                            ('E', E), ('F', F),('img_shape',self.img_shape)])

        cv2.destroyAllWindows()
        return camera_model
