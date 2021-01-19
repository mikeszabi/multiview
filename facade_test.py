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

print("Testing IO for meshes ...")
pcd = o3d.io.read_point_cloud(os.path.join(data_folder,'SamPointCloud.ply'))

coord_frames = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0,0,0])

#o3d.visualization.draw_geometries([pcd,coord_frames])


vis = o3d.visualization.Visualizer()

vis.create_window(window_name='pcd', visible=True)

vis.add_geometry(pcd)
vis.add_geometry(coord_frames)

#vis.add_geometry(pcd_images[0])

vis.get_render_option().point_size=3
vis.get_render_option().show_coordinate_frame=False

pcdtools.transform_to_origo(vis)

vis.update_geometry()
vis.poll_events()
vis.update_renderer()


vis.run()

vis.destroy_window()
