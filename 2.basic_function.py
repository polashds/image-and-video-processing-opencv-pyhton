import cv2 
import numpy as np 

# ## 1. Convert to gray scale

# img = cv2.imread('resources/lena.png')
# #cv2.cvtColor and COLOR_BGR2GRAY will convert to gray scale
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print("Color_img Shape: ", img.shape)
# print("img_gray Shape: ", img_gray.shape)
# cv2.imshow("Color_img", img)
# cv2.imshow("Gray_img", img_gray)
# cv2.waitKey(0)

# The benefit of gray scale is that it reduces the amount of data to process,
# which can speed up image processing tasks.

## Convert to blur 

# img = cv2.imread('resources/lena.png')
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # cv2.GaussianBlur will apply a Gaussian blur to the image
# # (7,7) is the kernel size, and 0 is the standard deviation in the X and Y directions.
# # A larger kernel size will result in a more blurred image.
# # The kernel size must be odd and positive.
# img_blur = cv2.GaussianBlur(img_gray, (7,7), 0)
# print("Color_img Shape: ", img.shape)
# print("img_gray Shape: ", img_gray.shape)
# cv2.imshow("Color_img", img)
# cv2.imshow("Gray_img", img_gray)
# cv2.imshow("img_blur", img_blur)
# cv2.waitKey(0)


### 3. Convert to cannyImg

img = cv2.imread('resources/lena.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (7,7), 0)
# cv2.Canny will apply the Canny edge detection algorithm to the image
# The two parameters are the lower and upper thresholds for the hysteresis procedure.
# The Canny edge detection algorithm is a multi-stage algorithm that detects a wide range of edges
# in images. It is widely used in computer vision and image processing.
# The first threshold is used to identify strong edges, and the second threshold is used to identify
# weak edges. If a weak edge is connected to a strong edge, it is considered an
img_canny = cv2.Canny(img_blur, 100,100)
print("Color_img Shape: ", img.shape)
print("img_gray Shape: ", img_gray.shape)
cv2.imshow("Color_img", img)
cv2.imshow("Gray_img", img_gray)
cv2.imshow("img_blur", img_blur)
cv2.imshow("img_canny", img_canny)
cv2.waitKey(0)
