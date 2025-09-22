import numpy as np
import imageio.v2 as imageio
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
from PIL import Image, ImageDraw, ImageFilter

img=cv2.imread('1_images_for_read_show_and_stop_image_from_directory/digit1.png')[:,:,0]
img = np.array(img)
print("Image " + str(1) + " shape is ", img.shape)

resizedImage = cv2.resize(img, dsize=(28, 28)) # Possbile resize options
resizedImage = np.array(resizedImage)
print("After resize image " + str(1) + " shape is ", resizedImage.shape)



reshapedImage = np.reshape(resizedImage, (1, 784))
print("After reshape image " + str(1) + " shape is ", reshapedImage.shape)
pd.DataFrame(reshapedImage).to_csv("2_images_covnerted_in_CSV/image" + str(1) + "_converted_image_in_one_row.csv")

reshapedImage = np.reshape(reshapedImage, (28, 28))
print("After agin image " + str(1) + " reshape the shape is ", reshapedImage.shape)
pd.DataFrame(reshapedImage).to_csv("2_images_covnerted_in_CSV/image" + str(1) + "_converted_in_28x28_csv.csv")

print("#####################################################")
print("Images coverted in .Csv files")