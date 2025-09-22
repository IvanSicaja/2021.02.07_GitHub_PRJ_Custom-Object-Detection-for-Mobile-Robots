import os
import cv2
import matplotlib.pyplot as plt
import numpy
import numpy as np

#Check file number in the directory
#
custom_test_images_directory_path= '1_images_for_read_show_and_stop_image_from_directory'
number_of_files_in_folder = len(os.listdir(custom_test_images_directory_path))
print("Number of images in diractory is: ",number_of_files_in_folder)
current_name= custom_test_images_directory_path+"/digit"+str(1)+".png"

img1 = cv2.imread(current_name)
cv2.imshow('sample image', img1)
print("img1 shape: ", img1.shape)

img2 = cv2.imread(current_name)
print("img2 shape: ", img2.shape)
plt.imshow(img2)
plt.show()

img3 = cv2.imread(current_name)[:,:,0]
print("img3 shape: ", img3.shape)
plt.imshow(img3,cmap=plt.cm.binary)
plt.show()

img4 = np.invert(np.array([img3]))[0,:,:]
print("img4 shape: ", img4.shape)
plt.imshow(img4,cmap=plt.cm.binary)
plt.show()






cv2.waitKey(0)  # waits until a key is pressed
cv2.destroyAllWindows()  # destroys the window showing image

