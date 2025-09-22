import os

# Just type "../" if you want to go to the parent directory
custom_test_images_directory_path = '../handwritten_digits.model'
number_of_testing_images = os.listdir(custom_test_images_directory_path)
print("Number of images in diractory is: ",number_of_testing_images)