import numpy as np
import imageio.v2 as imageio
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
from PIL import Image, ImageDraw, ImageFilter

# varify the path using getcwd()
cwd = os.getcwd()
# print the current directory
print("Current working directory is:", cwd)

delete_files_folder = os.listdir('2_images_covnerted_in_CSV')
print(len(delete_files_folder))
os.chdir(r"2_images_covnerted_in_CSV")


file_for_delete_1 = 'digit'+str(1)+'png'
os.remove(file_for_delete_1)

print("The directory is cleaned.")
print("#####################################################")

# varify the path using getcwd() 
cwd = os.getcwd()
# print the current directory 
print("Current working directory is:", cwd)

os.chdir(r"../..")
# varify the path using getcwd() 
cwd = os.getcwd()
# print the current directory 
print("Current working directory is:", cwd)

img = (imageio.imread("3.2_Extended_segmented_images/Extended_sgmented_character_" + str(i + 1) + ".jpg",as_gray=True))
img = np.array(img)

print("#####################################################")
print("Image " + str(i + 1) + " shape is ", img.shape)
print("Image " + str(i + 1) + " shape is ", img.shape)

resizedImage = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
resizedImage = np.array(resizedImage)

print("After resize image " + str(i + 1) + " shape is ", resizedImage.shape)

reshapedImage = np.reshape(resizedImage, (1, 784))

print("After reshape image " + str(i + 1) + " shape is ", reshapedImage.shape)

pd.DataFrame(reshapedImage).to_csv(
    "4_Segmented_images_converted_in_CSV/image_" + str(i + 1) + "_converted_image_in_one_row.csv")

reshapedImage = np.reshape(reshapedImage, (28, 28))

print("After agin image " + str(i + 1) + " reshape the shape is ", reshapedImage.shape)
print("#####################################################")

pd.DataFrame(reshapedImage).to_csv(
    "4_Segmented_images_converted_in_CSV/image_" + str(i + 1) + "_converted_in_28x28_csv.csv")



print("Images coverted in .Csv files")