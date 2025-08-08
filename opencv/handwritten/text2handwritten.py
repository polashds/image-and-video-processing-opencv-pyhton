from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

# Load text
text = "Python is great"

# Create a blank white image
img = Image.new('RGB', (800, 200), color='white')

# Load a handwriting font (.ttf file in the current directory)
font = ImageFont.truetype("handwriting.ttf", 40)

# Draw text on image
draw = ImageDraw.Draw(img)
draw.text((20, 50), text, font=font, fill='black')

# Save the image
img.save("new_text.png")

# Read and show image using OpenCV
hand_text = cv2.imread("new_text.png")
cv2.imshow("hand_text", hand_text)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Read and show image using PIL
hand_text = Image.open("new_text.png")
hand_text.show()