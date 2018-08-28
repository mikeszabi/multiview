# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 11:55:28 2018

@author: MikeSzabolcs
"""

import re
import os
import glob
import numpy as np
import cv2
import datetime


def buffer_from_file(buffer_file):
    buffer=None
    with open(buffer_file, "rb") as binary_file:
        # Read the whole file at once
        buffer = binary_file.read()
    return buffer

def im_from_buffer(buffer):
    buf=np.frombuffer(buffer, dtype=np.uint8, count=-1, offset=0)
    n_bytes=len(buf)
    height=4096 # number of rows
    
    if (n_bytes % height == 0) and (n_bytes>0):
      width=int(n_bytes/height) # number of cols
      im_tmp=buf.reshape((width,height))
      im_debayer = cv2.cvtColor(im_tmp, cv2.COLOR_BAYER_BG2BGR)
      im_rotated=cv2.transpose(im_debayer)
      im_flipped=cv2.flip( im_rotated, 1 )
    else:
      im_flipped=None  
    return im_flipped

base_dir=r'e:\IT Quality Services Kereskedelmi és Szolgáltató Kft\Kutatás-Fejlesztés - Dokumentumok\OneDrive - IT Quality Services Kereskedelmi és Szolgáltató Kft\EON\caronboardcalibration'
run_id='20180808_153220'
ext='.jpg'

raw_dir=os.path.join(base_dir,run_id)
raw_list=glob.glob(raw_dir+'/roi*')

script_path = os.getcwd()

for buffer_file in raw_list:
  #buffer_file=raw_list[0]
  print(buffer_file)    
  if not os.path.isfile(os.path.splitext(buffer_file)[0]+ext):
    #print('generating jpg')
    png_file=buffer_file+'.jpg'
    if not os.path.exists(png_file):
      buffer=buffer_from_file(buffer_file)
      im_debayer=im_from_buffer(buffer)
      if im_debayer is not None:    
        os.chdir(os.path.dirname(png_file))
        cv2.imwrite(os.path.basename(png_file), im_debayer, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        os.chdir(script_path)

      else:
        print('Buffer Problem')