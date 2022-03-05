# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 14:36:38 2022

@author: Trailblazers
"""

# # Draw a point


import math
import cv2
import matplotlib.pyplot as plt
import numpy as np

def draw_point(img, p, color) :
    cv2.circle(img, p, 2, color, 8)
    #cv2.circle( img, p, 2, color, cv2.cv.CV_FILLED, cv2.CV_AA, 0 )


#text file for points
txt_path = r'C:\Users\Mr-A\Desktop\points4.txt'
f = open(txt_path, 'w+')
###############################
# read image
img = cv2.imread(r'C:\Users\Mr-A\Desktop\Shape_1d_256i\AS\2337127678_158.png')

# convert image into RGB format
nemo = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# convert image into HSV format
hsv_nemo = cv2.cvtColor(nemo, cv2.COLOR_RGB2HSV)


# split image into h,s,v plane
h, s, v = cv2.split(hsv_nemo)
plt.subplot(1, 4, 1)
plt.imshow(img, cmap='gray')
plt.subplot(1, 4, 2)
plt.imshow(h, cmap='gray')
plt.subplot(1, 4, 3)
plt.imshow(s, cmap='gray')
plt.subplot(1, 4, 4)
plt.imshow(v, cmap='gray')
plt.show()

# threshold h plane
ret,thresh = cv2.threshold(h,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
hull = []
 
# calculate points for each contour
for i in range(len(contours)):
    # creating convex hull object for each contour
    hull.append(cv2.convexHull(contours[i], False))
print(hull)

# visualization of hull points
############################################
drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
 
# draw contours and hull points
for i in range(len(contours)):
    color_contours = (255, 255, 255) # green - color for contours
    color = (255, 0, 0) # blue - color for convex hull
    # draw ith contour
    cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
    # draw ith convex hull object
    cv2.drawContours(drawing, hull, i, color, 1, 8)
plt.figure(figsize=(8, 8));
plt.imshow(drawing[:,:,::-1])
plt.show()
# ###########################################
# # The extreme points

list_of_pts = [] 
for ctr in contours:
    list_of_pts += [pt[0] for pt in ctr]



# #################################################
# #creating list of extreme contours
class clockwise_angle_and_distance():

    def __init__(self, origin):
        self.origin = origin

    def __call__(self, point, refvec = [0, 1]):
        if self.origin is None:
            raise NameError("clockwise sorting needs an origin. Please set origin.")
        # Vector between point and the origin: v = p - o
        vector = [point[0]-self.origin[0], point[1]-self.origin[1]]
        # Length of vector: ||v||
        lenvector = np.linalg.norm(vector[0] - vector[1])
        # If length is zero there is no angle
        if lenvector == 0:
            return -math.pi, 0
        # Normalize vector: v/||v||
        normalized = [vector[0]/lenvector, vector[1]/lenvector]
        dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1] # x1*x2 + y1*y2
        diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1] # x1*y2 - y1*x2
        angle = math.atan2(diffprod, dotprod)
        # Negative angles represent counter-clockwise angles so we need to 
        # subtract them from 2*pi (360 degrees)
        if angle < 0:
            return 2*math.pi+angle, lenvector
        # I return first the angle because that's the primary sorting criterium
        # but if two vectors have the same angle then the shorter distance 
        # should come first.
        #print(lenvector)
        return angle, lenvector

center_pt = np.array(list_of_pts).mean(axis = 0) # get origin
#print(np.ceil(center_pt))
clock_ang_dist = clockwise_angle_and_distance(center_pt) # set origin
list_of_pts = sorted(list_of_pts, key=clock_ang_dist) # use to sort
# force the list of points into cv2 format and then
ctr = np.array(list_of_pts).reshape((-1,1,2)).astype(np.int32)
ctr = cv2.convexHull(ctr) # done.
#drawing sorted contours
cv2.drawContours(img,ctr,-1,(255,0,0),5)
plt.imshow(img)
    #cv2.rectangle(drawing,(127,127),(468,468),(0,255,0),-1)
# plt.imshow(drawing[:,:,::-1])
plt.show()


# #plotting all points
# ################################
for p in list_of_pts:
      draw_point(img, p, (0,0,255)) 
plt.imshow(img)
plt.show()
# #################################
# #left,top,right and bottom points
print(ctr)
l_m = tuple(ctr[ctr[:, :, 0].argmin()][0])
print('left=',l_m)
r_m = tuple(ctr[ctr[:, :, 0].argmax()][0])
print('right=',r_m)
t_m = tuple(ctr[ctr[:, :, 1].argmin()][0])
print('top=',t_m)
b_m = tuple(ctr[ctr[:, :, 1].argmax()][0])
print('bottom=',b_m)
# get a list of points
FourCorner_list = [l_m,r_m,t_m,b_m]
for p in FourCorner_list:
      draw_point(img, p, (0,255,0)) 
plt.imshow(img)
plt.show()


             