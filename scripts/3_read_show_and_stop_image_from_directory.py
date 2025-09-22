import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

#Check file number in the directory
#
custom_test_images_directory_path= '1_images_for_read_show_and_stop_image_from_directory'
number_of_files_in_folder = len(os.listdir(custom_test_images_directory_path))
print("Number of images in diractory is: ",number_of_files_in_folder)

for i in range(number_of_files_in_folder):
    current_name= custom_test_images_directory_path+"/digit"+str(i+1)+".png"
    img = cv2.imread(current_name)
    cv2.imshow('sample image', img)
    cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()  # destroys the window showing image

