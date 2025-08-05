import cv2 

# # Reading images

# img = cv2.imread("resources/lena.png")

# #print(img)
# print(img.shape)

# # Displaying images in console
# cv2.imshow("Output", img)
# # Wait for a key press and close the image window
# # cv2.waitKey(0) waits indefinitely until a key is pressed
# cv2.waitKey(0)

# # reading videos
# cap = cv2.VideoCapture("resources/elon.mp4")

# # this read the multiple frames of the video
# # and display them one by one
# while True:
#     # cap.read() this will return success and img 
#     success, img = cap.read()
#     print(img.shape)
#     cv2.imshow("Output", img)

#     # hexa number for q to quit
#     # 0xFF is used to mask the value returned by cv2.waitKey()
#     # ord('q') converts the character 'q' to its ASCII value
#     # if the key pressed is 'q', break the loop
#     # cv2.waitKey(1) waits for 1 millisecond
#     # if a key is pressed, it returns the ASCII value of the key
#     # if no key is pressed, it returns -1
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break 


## reading webcam

cap = cv2.VideoCapture(0)

cap.set(3, 640) # set width
cap.set(4, 480) # set height

# this read the multiple frames of the webcam
# and display them one by one
while True:
    success, img = cap.read()
    print(img.shape)
    cv2.imshow("Output", img)

    # without this line, the webcam will not close
    # cv2.waitKey(1) waits for 1 millisecond
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 