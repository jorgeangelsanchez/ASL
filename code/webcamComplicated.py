#!/usr/bin/env python
#coding: utf8

"""
Code originally by Brian R. Pauw and David Mannicke.
Modified by James Tompkin for Brown CSCI1430.

Initial Python coding and refactoring:
	Brian R. Pauw
With input from:
	Samuel Tardif

Windows compatibility resolution: 
	David Mannicke
	Chris Garvey

Windows compiled version:
	Joachim Kohlbrecher
"""

"""
Overview
========
This program uses OpenCV to capture images from the camera, Fourier transform
them and show the Fourier transformed image alongside the original on the screen.

$ ./liveFFT2.py

Required: A Python 3.x installation (tested on 3.5.3),
with: 
    - OpenCV (for camera reading)
    - numpy, matplotlib, scipy, argparse
"""

__author__ = "Brian R. Pauw, David Mannicke; modified for Brown CSCI1430 by James Tompkin"
__contact__ = "brian@stack.nl; james_tompkin@brown.edu"
__license__ = "GPLv3+"
__date__ = "2014/01/25; modifications 2017--2019"
__status__ = "v2.1"

#Necessary imports
import cv2 #opencv-based functions
import time
import math
import numpy as np
from scipy import ndimage
from skimage import io
from skimage import img_as_float, img_as_ubyte
from skimage.color import rgb2gray

#Initialize our values for later use
wn = "FD"
use_camera = True
im = 0
imJack = 0
phaseOffset = 0
rollOffset = 0

def __init__(self, **kwargs):

        # Camera device
        self.vc = cv2.VideoCapture(0)
        if not self.vc.isOpened():
            print( "No camera found or error opening camera; using a static image instead." )
            self.use_camera = False
        else:
            # We found a camera!
            # Requested camera size. This will be cropped square later on, e.g., 240 x 240
            ret = self.vc.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            ret = self.vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        # Set the size of the output window
        cv2.namedWindow(self.wn, 0)

        # Main loop
        while True:
            a = time.perf_counter()
            self.camimage_ft()
            print('framerate = {} fps \r'.format(1. / (time.perf_counter() - a)))

        def camimage_ft(self):
        
            if self.use_camera:
                # Read image
                rval, im = self.vc.read()
                # Convert to grayscale and crop to square
                # (not necessary as rectangular is fine; just easier for didactic reasons)
                im = img_as_float(rgb2gray(im))
                # Note: some cameras across the class are returning different image sizes
                # on first read and later on. So, let's just recompute the crop constantly.
            
                if im.shape[1] > im.shape[0]:
                    cropx = int((im.shape[1]-im.shape[0])/2)
                    cropy = 0
                elif im.shape[0] > im.shape[1]:
                    cropx = 0
                    cropy = int((im.shape[0]-im.shape[1])/2)

                self.im = im[cropy:im.shape[0]-cropy, cropx:im.shape[1]-cropx]

            # Set size
            width = self.im.shape[1]
            height = self.im.shape[0]
            cv2.resizeWindow(self.wn, width*2, height*2)

            # This code reads an image from your webcam. If you have no webcam, e.g.,
            # a department machine, then it will use a picture of an intrepid TA.
            #
            # Output image visualization:
            # Top left: input image
            # Bottom left: amplitude image of Fourier decomposition
            # Bottom right: phase image of Fourier decomposition
            # Top right: reconstruction of image from Fourier domain
            #
            # Let's start by peforming the 2D fast Fourier decomposition operation
            imFFT = np.fft.fft2( self.im )
        
            # Then creating our amplitude and phase images
            amplitude = np.sqrt( np.power( imFFT.real, 2 ) + np.power( imFFT.imag, 2 ) )
            phase = np.arctan2( imFFT.imag, imFFT.real )
        
            # We will reconstruct the image from this decomposition later on (far below at line 260); have a look now.

            ###########################################################
            # # Just the central dot
            #amplitude = np.zeros( self.im.shape )
            # # a = np.fft.fftshift(amplitude)
        
            # # NOTE: [0,0] here is the 0-th frequency component around which all other frequencies oscillate
            #amplitude[0,0] = 40000
        
            # # amplitude = np.fft.fftshift(a)
            
