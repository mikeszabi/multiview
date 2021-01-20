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
import imtools_msz as imsz
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

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH    = 6

def init_feature(name,n_feature_point=None):
    if n_feature_point is None:
        sift_default=1000
        surf_default=1000
        orb_default=500
    else:
        sift_default=n_feature_point
        surf_default=n_feature_point
        orb_default=n_feature_point        
        
    chunks = name.split('-')
    if chunks[0] == 'sift':
        detector = cv2.xfeatures2d.SIFT_create(sift_default)
        norm = cv2.NORM_L2
    elif chunks[0] == 'surf':
        detector = cv2.xfeatures2d.SURF_create(surf_default)
        norm = cv2.NORM_L2
    elif chunks[0] == 'orb':
        detector = cv2.ORB_create(orb_default)
        norm = cv2.NORM_HAMMING
    elif chunks[0] == 'akaze':
        detector = cv2.AKAZE_create()
        norm = cv2.NORM_HAMMING
    elif chunks[0] == 'brisk':
        detector = cv2.BRISK_create()
        norm = cv2.NORM_HAMMING
    else:
        return None, None
    if 'flann' in chunks:
        if norm == cv2.NORM_L2:
            flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        else:
            flann_params= dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 6, # 12
                               key_size = 12,     # 20
                               multi_probe_level = 1) #2
        matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    else:
        matcher = cv2.BFMatcher(norm)
    return detector, matcher

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape[:2]
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1[0]),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2[0]),5,color,-1)
    return img1,img2


def get_strength(M):
    # strength of homogrpahy - works only if planar surfaces are present
    scale=np.abs((M[0,0]+M[1,1])/2)
    M_strength=np.abs(1-np.linalg.det(M))
    # if M_strength>1:
    #     M_strength=1
    return M_strength/scale
    
    
class homography_chain:
    def __init__(self,feat='orb',ratio=1,image_max_dim=1014,n_feature_point=1000):
        # model specific parameters
     
        self.image_max_dim=image_max_dim
        
        self.detector,self.matcher=init_feature(feat,n_feature_point=n_feature_point)

        self.ratio=ratio
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

        ## extract the matched keypoints
        src_pts  = np.float32([self.feats[im1][0][m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts  = np.float32([self.feats[im2][0][m.trainIdx].pt for m in matches]).reshape(-1,1,2)

        ## find homography matrix and do perspective transform
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        return M

    
    def create_features(self):
        print('CREATING FEATURES')
        for im in self.im_list:
            print(os.path.basename(im))
            img = cv2.imread(im,0)  #queryimage # left image
            img, scale=imsz.imRescaleMaxDim(img, self.image_max_dim, boUpscale = False, interpolation = 1)

            kp, des = self.detector.detectAndCompute(img,None)
            #des=des.astype('float32')
            self.feats[im]=(kp,des)
            
        return 
    
    def match_features(self,mode='next'):
        print('CREATING MATCHES')
        
        # min_strength=1
        self.initial=0
       
        for i1 in range(0,self.n_im):
            im1=self.im_list[i1]
            if mode=='next':
                if i1<self.n_im-1:
                    compare_list=[i1+1]
                else:
                    compare_list=[]
            else:
                #compare all
                compare_list=range(i1+1,self.n_im)
            for i2 in compare_list:
                im2=self.im_list[i2]
                matches = self.matcher.knnMatch(self.feats[im1][1],self.feats[im2][1],k=2)
                
                # ratio test
                good = []
                for m in matches:
                    if len(m)<2:
                        print(m)
                    else:
                        if m[0].distance < self.ratio*m[1].distance:
                            good.append(m[0])
                
                ######
                dmatches = sorted(good, key = lambda x:x.distance)
                self.match_graph.add_edge(i1,i2,length=float("{0:d}".format(len(dmatches))))


                # if len(dmatches)>10:
                #     self.Ms[im1,im2]=self.create_homography(dmatches, im1, im2)
                    
                #     if self.Ms[im1,im2] is not None:
                #         M_strength=get_strength(self.Ms[im1,im2])
                #     else:
                #         M_strength=0
                #     print('{} - {} : {}'.format(i1,i2, M_strength))
                #     if M_strength<min_strength:
                #         min_strength=M_strength
                #         self.initial=i1
    
                #     self.match_graph.add_edge(i1,i2,length=float("{0:.3f}".format(M_strength)))

                # else:
                #     self.match_graph.add_edge(i1,i2,length=float("{0:.3f}".format(1000)))
            
       
        return