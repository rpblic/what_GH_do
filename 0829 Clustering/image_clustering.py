import matplotlib.image as mpimg

image_file= "filename"
img= mpimg.imread(image_file)

top_row= img[0]
top_left_pixel=top_row[0]
red, green, blue= top_left_pixel
