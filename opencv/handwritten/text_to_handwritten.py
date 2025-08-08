from PIL import Image, ImageDraw, ImageFont
import cv2
import os

# Path to font file
font_path = "handwriting.ttf"

# Check if font file exists
if not os.path.exists(font_path):
    raise FileNotFoundError(f"Font file not found: {font_path}")

# Create blank white image
img = Image.new('RGB', (800, 200), color='white')

# Load the handwriting font
font = ImageFont.truetype(font_path, 40)

# Draw text on the image
draw = ImageDraw.Draw(img)
draw.text((20, 50), "Python is great", font=font, fill='black')

# Save to file
output_path = "new_text.png"
img.save(output_path)

# Show with OpenCV
image = cv2.imread(output_path)
cv2.imshow("Handwritten Text", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
