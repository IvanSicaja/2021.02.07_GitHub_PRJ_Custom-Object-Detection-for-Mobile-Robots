import cv2
import numpy as np

path= '1_images_for_read_show_and_stop_image_from_directory/H.png'  # Wanted image
img = cv2.imread(path)
print(img.shape)

imgResize = cv2.resize(img,(1000,500))
print(imgResize.shape)

imgCropped = img[46:119,352:495]
#print(imgCropped)

cv2.imshow("Image",img)
#cv2.imshow("Image Resize",imgResize)
cv2.imshow("Image Cropped",imgCropped)

cv2.waitKey(0)