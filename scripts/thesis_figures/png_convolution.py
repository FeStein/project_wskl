"""
File: png_convolution.py
Author: Felix Steinmetz
Email: fsteinme@rhrk.uni-kl.de / felix.steinmetz@gmx.de
Github: https://github.com/FeStein
Description: Takes an input image as an argument (-f) converts it to greyscale
and saves the images with 3 applied convolution filters
(horizontal,vertical,egde) as .png
"""

import cv2 
import numpy as np

import argparse                                                                                                                                          
                                                                                                                                                           
ap = argparse.ArgumentParser()                                                                                                                           
ap.add_argument("-f", "--filename", required=False, type=str, default='img.jpeg',                                                                                         
                help="path to file")                                                                                                                     
                                                                                                                                                          
args = vars(ap.parse_args())                                                                                                                             
filename = args["filename"] 

#define convolution filters/kernels
horioz_conv = np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]])
vertical_conv = np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]])
edge_conv = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])

#read image and convert to grey
image = cv2.imread(filename)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#apply the convolution
hout = cv2.filter2D(gray, -1, horioz_conv)
vout = cv2.filter2D(gray, -1, vertical_conv)
eout = cv2.filter2D(gray, -1, edge_conv)

#write out the files (add original in greyscale)
cv2.imwrite("original.png", gray)
cv2.imwrite("horizontal_conv.png", hout)
cv2.imwrite("vertical_conv.png", vout)
cv2.imwrite("edge_conv.png", eout)
