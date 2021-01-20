# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 14:25:32 2018

@author: szabo
"""

import importlib
import os
import imtools_msz as imsz
import matplotlib.pyplot as plt
import cv2


import homography_chain as hc

importlib.reload(hc)

import tsp_visualization as tspv
importlib.reload(tspv)


base_dir=r'd:\DATA\EON_LOCAL\SEQUENCES\20200304_141400\276422'
#run_id='20180907_103336_sel1'
run_id='Master'
ext='.jpg'
image_max_dim=1014


im_dir=os.path.join(base_dir,run_id)
jpg_list=imsz.imagelist_in_depth(im_dir,level=1)

hcobj=hc.homography_chain(feat='orb',ratio=0.75,image_max_dim=image_max_dim,n_feature_point=1000)
hcobj.add_image_list(jpg_list)

hcobj.create_features()
hcobj.match_features()

G=hcobj.match_graph

plt.figure(10)
pos = tspv.DrawGraph(G,'black')
opGraph = tspv.christofedes(G, pos)
plt.figure(11)
pos1 = tspv.DrawGraph(opGraph,'r') 
plt.show()

## visualize pairwise homography
i1=0
i2=1
im1=jpg_list[i1]
im2=jpg_list[i2]

img1 = cv2.imread(im1,0)  #queryimage # left image
img2 = cv2.imread(im2,0)  #queryimage # left image
image_max_dim=hcobj.image_max_dim
img1, scale = imsz.imRescaleMaxDim(img1, image_max_dim, boUpscale = False, interpolation = 1)
img2, scale = imsz.imRescaleMaxDim(img2, image_max_dim, boUpscale = False, interpolation = 1)

M=hcobj.Ms[im1,im2]

img1_2 = cv2.warpPerspective(img1, M, (img2.shape[1],img2.shape[0]))

fig=plt.figure(12)

axarr = fig.subplots(1, 3)
axarr[0].imshow(img1,cmap='gray')
axarr[1].imshow(img1_2,cmap='gray')
axarr[2].imshow(img2,cmap='gray')
