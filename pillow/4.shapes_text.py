import cv2
import numpy as np

# zero matrix generation and black window
img = np.zeros((512, 512, 3), np.uint8)
print(img.shape)

# img[:] = 255,0,0  blue color window
img[:] = 255,0,0 #BGR 0,0,0

# img[:] = 0,255,0  blue color window
img[:] = 0,255,0 #BGR 0,0,0

# img[:] = 0,0,255  red color window
img[:] = 0,0,255 #BGR 0,0,0

# img[:] = 10,155,255  different color window with combination of blue green and red
img[:] = 0,0,255 #BGR 0,0,0

# Create a line
# cv2.line(img, (0,0), (300,400), (0,255,0), 3)

### Rectangle
# cv2.rectangle(img, (0,0), (250, 350), (0,0,255), 7)

### Create circle 
# cv2.circle(img, (400,50), 50, (0,0,255), 4)

### Put texts
cv2.putText(img, "codon ananlytics", (200,440), cv2.FONT_HERSHEY_COMPLEX,1, (0,255,0), 1)

cv2.imshow('Image', img)
cv2.waitKey(0)