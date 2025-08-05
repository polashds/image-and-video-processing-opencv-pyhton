import cv2
import numpy as np 

## Resizing images

# img = cv2.imread('Resources/lambo.png')
# print(img.shape)
# #resize take three arguments (height, width, channels) 
# resized_img = cv2.resize(img, (300, 200))
# print(resized_img.shape)
# cv2.imshow('Output', img)
# cv2.imshow('Resized_Output', resized_img)
# cv2.waitKey(0)


### Croping Image
img = cv2.imread('Resources/lambo.png')

#Crops a portion of the image using slicing.
#img[y1:y2, x1:x2] → This selects a rectangle from (x=200, y=0) to (x=500, y=200).
#You're effectively grabbing a region 300 pixels wide and 200 pixels tall from the top-middle portion of the image.

crop_img = img[0:200,200:500]
cv2.imshow('Output', img)
cv2.imshow('Crop_Output', crop_img)
cv2.waitKey(0)


