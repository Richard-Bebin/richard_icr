#app
 -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 20:22:28 2018

@author: NI6167
"""
import math
import cv2
import numpy as np
import os
import imutils
from scipy import ndimage

#edge
def find_countours(image):
    blur = cv2.GaussianBlur(image,(5,5),0)
    edged = cv2.Canny(blur, 50, 150)#, apertureSize=5)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    screenCnt = None
    
    for c in cnts:
        	# approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.015 * peri, True)
            
            # if our approximated contour has four points, then
            # we can assume that we have found our screen
            if len(approx) == 4:
                screenCnt = approx
                break
            
    if (screenCnt==None):
        return (0,0)
        
    pts = screenCnt.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = "float32")
     
    # the top-left point has the smallest sum whereas the
    # bottom-right has the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
 
    # compute the difference between the points -- the top-right
    # will have the minumum difference and the bottom-left will
    # have the maximum difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    area=cv2.contourArea(screenCnt)
    return(rect,area)

def correct_perspective(img,rect):
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    
    # ...and now for the height of our new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    
    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    
    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    dst = np.array([
            [0, 0],	[maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
    
    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (2200, 1700))
    return(warp)

def tilt_angle(image):
    
    kernel = np.ones((5,5),np.uint8)
    #_,thresh = cv2.threshold(img, 150, 255,cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(image,(5,5),0)
    img_edges = cv2.Canny(blur, 50, 150)#, apertureSize=5)
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 300, minLineLength=500, maxLineGap=10)
    #print(len(lines))
    angles = []
    line_len=[]
    

    for line in lines:
        x1, y1, x2, y2=line[0]
        if x1==x2:
            continue
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)
        line_len.append(x2-x1)
    
   
    mean_angle = np.mean(angles)
    return(mean_angle)


path='C://Users//NI6167//Documents//image1'
tpath='C://Users//NI6167//Documents//pro_image2'

files=os.listdir(path)

for file in files:
    image_name=path+'//'+file
    img=cv2.imread(image_name,0)
    rect,area=find_countours(img)
    #print(file)
    #print(rect,area)
    h,w=img.shape
    thresh_area=(h/2)*(w/2)
    
    if area>thresh_area:
        perspected=correct_perspective(img,rect)
        #print(cv2.isContourConvex(rect))
    else:
        perspected=img
    
    cv2.imwrite()    
    
    tilted_angle=tilt_angle(perspected)
    img_rotated = ndimage.rotate(perspected,tilted_angle)
    
    tar_file=tpath+'//'+file
    cv2.imwrite(tar_file,img_rotated) 
    

































    
#test area
    #i3=cv2.drawContours(im2, [screenCnt], -1, (0, 255, 0), 3)
    #plt.imshow(im2)

#i4=im2[488:2419,377:2750]
 
# now that we have our rectangle of points, let's compute
# the width of our new image
(tl, tr, br, bl) = rect
widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
 
# ...and now for the height of our new image
heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
 
# take the maximum of the width and height values to reach
# our final dimensions
maxWidth = max(int(widthA), int(widthB))
maxHeight = max(int(heightA), int(heightB))
 
# construct our destination points which will be used to
# map the screen to a top-down, "birds eye" view
dst = np.array([
	[0, 0],
	[maxWidth - 1, 0],
	[maxWidth - 1, maxHeight - 1],
	[0, maxHeight - 1]], dtype = "float32")
 
# calculate the perspective transform matrix and warp
# the perspective to grab the screen
M = cv2.getPerspectiveTransform(rect, dst)
warp = cv2.warpPerspective(im2, M, (2200, 1700))

cv2.imwrite('th3.jpg',warp) 

warp = cv2.warpPerspective(img1, M, (872, 656))


