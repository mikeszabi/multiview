# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 14:25:32 2018

@author: szabo
"""
import networkx as nx

import importlib
import os
import imtools_msz as imsz
import matplotlib.pyplot as plt
import cv2


import homography_chain as hc

importlib.reload(hc)

import tsp_visualization as tspv
importlib.reload(tspv)


#base_dir=r'd:\DATA\EON_LOCAL\SEQUENCES\20200304_141400\276422'
base_dir=r'd:\DATA\EON\Tests\bbox_from_annot_2'
#run_id='20180907_103336_sel1'
# run_id='Master'
run_id='image'
ext='.jpg'
image_max_dim=1014


im_dir=os.path.join(base_dir,run_id)
jpg_list=imsz.imagelist_in_depth(im_dir,level=1)

jpg_list=jpg_list[0:50]

hcobj=hc.homography_chain(feat='orb',ratio=0.75,image_max_dim=image_max_dim,n_feature_point=500)
hcobj.add_image_list(jpg_list)

hcobj.create_features()
hcobj.match_features(mode='next')
# ToDo: filter by pole bbox

G=hcobj.match_graph

# plt.figure(10)
# pos = nx.spring_layout(G)
# colors = [100]*25
# colors = '#A0CBE2'
# edge_labels = nx.get_edge_attributes(G,'length')
# nx.draw(G, pos, node_color='#A0CBE2', edge_color=colors,
#         edge_labels = edge_labels,
#         width=4, edge_cmap=plt.cm.Blues, with_labels=True)
# plt.show()

length_tsh=25
plt.figure(11)
pos1 = tspv.DrawGraph(G,edge_max=200,edge_color='r') 
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



fig=plt.figure(12)

axarr = fig.subplots(1, 2)
axarr[0].imshow(img1,cmap='gray')
axarr[1].imshow(img2,cmap='gray')
