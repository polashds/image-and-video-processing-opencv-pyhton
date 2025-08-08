from PIL import ImageDraw 
from PIL import Image 
from PIL import ImageFont 

# Pass link to your image location 
pic = Image.open('E:\CV-Projects-Vault\image-and-video-processing-py-cv\resources\car1.jpeg')

# make a copy of the pic 
drawing = ImageDraw.Draw(pic) 

# fill color for the font 
fill_color =(255,250,250) 

# watermark font 
font = ImageFont.truetype("arial.ttf", 60) 

# Watermark position 
position = (0, 0) 

# Writing text to picture 
drawing.text(position, text='Lion is King', 
fill=fill_color, font=font) 
pic.show() 

# saving image 
pic.save('watermarkedimg.jpg') 