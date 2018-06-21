# -*- coding: utf-8 -*-
"""
Created on Mon May  8 21:56:28 2017

@author: SzMike
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread(r'e:\Pictures\TestSets\TestSequencies\ComPair\3_zoomedin.jpg',0)          # queryImage
img2 = cv2.imread(r'e:\Pictures\TestSets\TestSequencies\ComPair\3_zoomedout.jpg',0)          # queryImage

%matplotlib qt5
# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
-plt.figure()
img3=np.zeros(img2.shape)
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:50], img3, flags=2)

#plt.imshow(img3),plt.show()

img11=np.zeros(img2.shape)

img11 = cv2.drawKeypoints(img1,kp1,img11, color=(0,255,0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
plt.imshow(img11),plt.show()

plt.figure()
img21=np.zeros(img2.shape)

img21 = cv2.drawKeypoints(img2,kp2,img21, color=(0,255,0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
plt.imshow(img21),plt.show()