# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 21:25:39 2018

@author: szabo
"""

import os
import cv2
import numpy as np
#from matplotlib import pyplot as plt
#import re
#from collections import defaultdict
import networkx as nx

#http://benalexkeen.com/implementing-djikstras-shortest-path-algorithm-with-python/
# https://towardsdatascience.com/getting-started-with-graph-analysis-in-python-with-pandas-and-networkx-5e2d2f82f18e

#class Graph():
#    def __init__(self):
#        """
#        self.edges is a dict of all possible next nodes
#        e.g. {'X': ['A', 'B', 'C', 'E'], ...}
#        self.weights has all the weights between two nodes,
#        with the two nodes as a tuple as the key
#        e.g. {('X', 'A'): 7, ('X', 'B'): 2, ...}
#        """
#        self.edges = defaultdict(list)
#        self.weights = {}
#    
#    def add_edge(self, from_node, to_node, weight):
#        # Note: assumes edges are bi-directional
#        self.edges[from_node].append(to_node)
#        self.edges[to_node].append(from_node)
#        self.weights[(from_node, to_node)] = weight
#        self.weights[(to_node, from_node)] = weight

def get_strength(M):
    M_strength=np.abs(1-np.linalg.det(M))
    if M_strength>1:
        M_strength=1
    return M_strength
    
    
class homography_chain:
    def __init__(self):
        # model specific parameters
     
        self.orb = cv2.ORB_create(10000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.feats={}
        self.matches={}
#        self.img_width=1000
        self.im_list=None
        self.n_im=0
        self.Ms={}
        self.match_graph=nx.Graph()
        
       
    def add_image_list(self,im_list=None):
        # Must be called
        # Todo make sure im_list is valid
        if im_list:
            self.im_list=im_list
            self.n_im=len(im_list)

    
    def create_homography(self, matches, im1, im2):
        dmatches = sorted(matches, key = lambda x:x.distance)

        ## extract the matched keypoints
        src_pts  = np.float32([self.feats[im1][0][m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
        dst_pts  = np.float32([self.feats[im2][0][m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)


        ## find homography matrix and do perspective transform
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        return M

    
    def create_features(self):
        print('CREATING FEATURES')
        for im in self.im_list:
            print(os.path.basename(im))
            img = cv2.imread(im,0)  #queryimage # left image
            height, width = img.shape[:2]

            #img = cv2.resize(img,(int(0.5*width), int(0.5*height)), interpolation = cv2.INTER_CUBIC)

            # ToDO:check if valid image
            # Resize image
#            height, width = img.shape[:2]
#            img = cv2.resize(img,(int(0.5*width), int(0.5*height)), interpolation = cv2.INTER_CUBIC)
#            
            self.feats[im] = self.orb.detectAndCompute(img,None)
            
        return 
    
    def match_features(self):
        print('CREATING MATCHES')
        
        min_strength=1
        self.initial=0
       
        for i1 in range(0,self.n_im):
            im1=self.im_list[i1]
            for i2 in range(i1+1,self.n_im):
                im2=self.im_list[i2]
                matches = self.bf.match(self.feats[im1][1],self.feats[im2][1])
                
                self.Ms[im1,im2]=self.create_homography(matches, im1, im2)
                
                M_strength=get_strength(self.Ms[im1,im2])
                print('{} - {} : {}'.format(i1,i2, M_strength))
                if M_strength<min_strength:
                    min_strength=M_strength
                    self.initial=i1
                
                self.match_graph.add_edge(i1,i2,length=float("{0:.3f}".format(M_strength)))
       
        return