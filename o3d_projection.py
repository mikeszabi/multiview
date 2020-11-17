# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 18:07:20 2020

@author: szabo
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 14:06:37 2019

@author: szabo
"""

import open3d as o3d
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import imtools_msz as imsz
import pcdtools

max_dim=1386

im_id='IMG00000000000000000000.png'

imagePaths=[r'e:\DATA\BRAUN\20201019_test_images\camera_1\IMG00000000000000000000.png',
            r'e:\DATA\BRAUN\20201019_test_images\camera_2\IMG00000000000000000000.png']
            


# grab the paths to the input images and initialize our images list
print("[INFO] loading images...")
# imagePaths = sorted(list(paths.list_images(args["images"])))
images = []
# loop over the image paths, load each one, and add them to our
# images to stich list
for imagePath in imagePaths:
    img = cv2.imread(imagePath)
    img, scale = imsz.imRescaleMaxDim(img, max_dim, boUpscale = False, interpolation = 1)
    images.append(img)

fs=[None]*4
cxs=[None]*4
cys=[None]*4

fs[0]=5124*scale
cxs[0]=2740*scale
cys[0]=1873*scale

fs[1]=5144*scale
cxs[1]=2720*scale
cys[1]=1895*scale

####
pcd_images=[None]*4
mesh_frames=[None]*4

###### create point clouds
z_image=3500
 
frame=images[0]  
f=fs[0]
cx=cxs[0]
cy=cys[0]

pcd_images[0]=pcdtools.pcd_from_planarimage(frame,f,cx,cy,z_image=3500)

z_image=3500
 
frame=images[1]  
f=fs[1]
cx=cxs[1]
cy=cys[1]


pcd_images[1]=pcdtools.pcd_from_planarimage(frame,f,cx,cy,z_image=3500)

#### create origo

mesh_frames[0] = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0,0,0])
mesh_frames[1] = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[60,0,0])

###

vis = o3d.visualization.Visualizer()

vis.create_window(window_name='pcd', visible=True)

vis.add_geometry(mesh_frames[0])
vis.add_geometry(mesh_frames[1])

vis.add_geometry(pcd_images[0])

vis.get_render_option().point_size=3
vis.get_render_option().show_coordinate_frame=False

pcdtools.transform_to_origo(vis)

vis.update_geometry()
vis.poll_events()
vis.update_renderer()


vis.run()

vis.destroy_window()





