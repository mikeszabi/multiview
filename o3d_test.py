# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 14:18:56 2019

@author: szabo
"""

import open3d as o3d
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

import Proto_2.scripts.tools.transformationtools as transformationtools
import Proto_2.scripts.tools.pcdtools as pcdtools

def process_points(cur_data):
    xyzr=np.zeros((len(cur_data.frame.pointClouds)*100,4),dtype=np.float64)
    i=-1
    for pcs in cur_data.frame.pointClouds:
        for points in pcs.points:
            i+=1
            xyzr[i,:]=np.asarray((points.y,points.z,points.x,points.reflectivity))
    xyz=xyzr[:,:3]
    pcd = o3d.geometry.PointCloud()

    # transform to image plane
    pcd.points = o3d.utility.Vector3dVector(xyz)
    #pcd.paint_uniform_color([1,0,0])
    #Frame mode
    pcd.transform([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    return pcd

###### parameters

th_size=(760,1014)
Lidar_FOV=38.4


# get intrinsics
#device='Master'
#calib_file=os.path.join(r'.\parameters','calib_mtx_'+device+'.npy')
#
#
#calib_mtx=np.load(calib_file)

# extrinsic lidar-cam
#angles=[1*np.pi/180,-3*np.pi/180,1*np.pi/180]
extrinsic_angles=[0,0,0]


vispcd = pcdtools.VisualizePCD(width=1024,height=720,fov=90)


pcd=process_points(cur_data)
vispcd.refresh_pcd(pcd)
vispcd.set_coordinate_and_update(yaw=-60*np.pi/180,dz=5000,dx=2000)

vispcd.vis.run() 

del vispcd

fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
#ax=fig.add_axes()
ax.set_ylabel('points')
ax.set_xlabel('distanc')
ax.set_title('distance histogram')

ax = fig.add_subplot(111)
#ax=fig.add_axes()
ax.set_ylabel('points')
ax.set_xlabel('distanc')
ax.set_title('distance histogram')
ax.plot(bins[1:],vals)
fig.show()