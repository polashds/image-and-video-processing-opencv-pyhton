import pywhatkit 
import cv2 as cv 

# text to convert to handwriting 
text = 'Python is great' 

# converting text to handwriting and saving as image 
pywhatkit.text_to_handwriting(text, 
save_to='new_text.png') 

# reaading image using cv 
hand_text = cv.imread("new_text.png") 
cv.imshow("hand_text", hand_text) 
cv.waitKey(0) 
cv.destroyAllWindows() 