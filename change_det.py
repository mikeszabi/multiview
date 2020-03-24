# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:28:55 2020

@author: szabo
"""

# run after homography

fig=plt.figure(1)
axarr = fig.subplots(1, 1)
axarr.imshow(img2,cmap='gray')

fig=plt.figure(3)
axarr = fig.subplots(1, 1)
axarr.imshow(img,cmap='jet')

im_diff=np.abs(img2.astype('float')-img1_2.astype('float'))

im_diffn=imsz.normalize(im_diff,vis_diag=False,ax=None,fig='')

im_diffn[img1_2==0]=0

fig=plt.figure(3)
axarr = fig.subplots(1, 1)
axarr.imshow(im_diffn,cmap='jet')