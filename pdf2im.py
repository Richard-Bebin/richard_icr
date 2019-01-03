# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
#np.set_printoptions(threshold=np.nan)
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import imutils
import os
path=r'C://Users//NI6167//Documents//icr'
os.chdir(path)
files=os.listdir(path)

from pdf2image import convert_from_path

for file in files:
    file_name=path+'//'+file
    images = convert_from_path(file_name,path)
    
    
    
import easyPDFConverter
converter = easyPDFConverter.PDFConverter()
try:
   pdf2image = converter.getPDF2Image()
   pdf2image.Convert("c:\\test\\input.pdf", "c:\\test\\output.jpg", "", -1, -1)
except easyPDFConverter.PDFConverterException as ex:
   print(ex)
   
im1 =cv2.imread('C://Users//NI6167//Documents//Benefits-Enrollment-Form-2.jpg',0)
#gray = cv2.cvtColor(im1, 6)
#gray = cv2.bitwise_not(im1)
im1.shape
#cv2.imwrite('grayed.jpg',im1)

kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(im1,kernel,iterations = 1)
cv2.imwrite('eroded.jpg',erosion)

dilation = cv2.dilate(erosion,kernel,iterations = 1)
cv2.imwrite('dilated.jpg',dilation)

M = np.float32([[1, 0, -50], [0, 1, -90]])

shifted = cv2.warpAffine(im1, M, (im1.shape[1], im1.shape[0]))
cv2.imshow("Shifted Up and Left", shifted)


#angle correction
thresh = cv2.threshold(im1, 0, 255,cv2.THRESH_BINARY_INV| cv2.THRESH_OTSU)[1]
cv2.imwrite('thresh.jpg',thresh)
        
coords = np.column_stack(np.where(thresh > 0))
angle = cv2.minAreaRect(coords)[-1]
 
# the `cv2.minAreaRect` function returns values in the
# range [-90, 0); as the rectangle rotates clockwise the
# returned angle trends to 0 -- in this special case we
# need to add 90 degrees to the angle
if angle < -45:
	angle = -(90 + angle)
 
# otherwise, just take the inverse of the angle to make
# it positive
else:
	angle = -angle
    
(h, w) = im1.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(im1, M, (w, h),
	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
cv2.imwrite('rotated3.jpg',rotated)

img=cv2.imread('C://Users//NI6167//Documents//icr//rotated1.jpg',0)
    