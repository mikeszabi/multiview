# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 21:02:20 2021

@author: szabo
"""

import open3d as o3d
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import imtools_msz as imsz
import pcdtools

data_folder=r'e:\DATA\FACADE'
data_folder=r'/mnt/data/FACADE/Szabinak'


#def custom_draw_geometry(pcd):
#    # The following code achieves the same effect as:
#    # o3d.visualization.draw_geometries([pcd])
#    vis = o3d.visualization.Visualizer()
#    vis.create_window()
#    vis.add_geometry(pcd)
#    vis.run()
#    vis.destroy_window()

#print("Testing IO for meshes ...")
pcd = o3d.io.read_point_cloud(os.path.join(data_folder,'SamPointCloud.ply'))
#
#
#
coord_frames = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0,0,0])

#o3d.visualization.draw_geometries([pcd,coord_frames])
#
#
vis = o3d.visualization.Visualizer()

vis.create_window(window_name='pcd', visible=True)

vis.add_geometry(pcd)
vis.add_geometry(coord_frames)


##############
img_shape=[2000,2000]
w,h=img_shape
pixel_size=1.5 # um
f=2 # mm
l=100 # m
bbox=[[400,400],
    [400,500],
    [500,500],
    [500,400]]


points=[[0,0,0]]
for bb in bbox:
    points.append([int(l*(bb[0]-w/2)*pixel_size/(f*1000)),int(l*(bb[1]-w/2)*pixel_size/(f*1000)),l])


lines = [[0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 2],
        [2, 3],
        [3, 4],
        [4,1 ]]

colors = [[1, 0, 0] for i in range(len(lines))]
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines),
)
line_set.colors = o3d.utility.Vector3dVector(colors)
vis.add_geometry(line_set)


#
#camera=o3d.camera(()
#
#ctr = vis.get_view_control()
#parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera_2021-01-20-14-38-53.json")
#ctr.convert_from_pinhole_camera_parameters(parameters)




vis.run()

#vis.destroy_window()

