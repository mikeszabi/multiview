# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 14:25:32 2018

@author: szabo
"""

import importlib
import os
import imtools_msz as imsz
import matplotlib.pyplot as plt


import homography_chain as hc
importlib.reload(hc)

import tsp_visualization as tspv
importlib.reload(tspv)



base_dir=r'd:\DATA\EON1\photosFor3D'
run_id='SceauxCastle\images'
ext='.jpg'

im_dir=os.path.join(base_dir,run_id)
jpg_list=imsz.imagelist_in_depth(im_dir,level=1)

hcobj=hc.homography_chain()
hcobj.add_image_list(jpg_list)

hcobj.create_features()
hcobj.match_features()

G=hcobj.match_graph

plt.figure(1)
pos = tspv.DrawGraph(G,'black')
opGraph = tspv.christofedes(G, pos)
plt.figure(2)
pos1 = tspv.DrawGraph(opGraph,'r') 
plt.show()

