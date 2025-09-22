import os,shutil
import cv2
from keras.preprocessing.image import ImageDataGenerator

# ---------> Define data augmentation properties <--------#
datagen = ImageDataGenerator(
        rotation_range=25,              # Range int [0-45] = rotation degrees NOTE: If value = 45, image rotation goes in the both directions with different values between 0-defined value
        width_shift_range=0.1,          # Range float [0-1] = %  NOTE: If value = 0.2, image goes up and down with different step values which are in the range between 0-defined value
        height_shift_range=0.1,         # Range float [0-1] = %  NOTE: If value = 0.2, image goes left and right with different step values which are in the range between 0-defined value
        shear_range=0.0,                # ? Range float [0-1] = %
        zoom_range=0.2,                 # Range float [0-1] = %
        horizontal_flip=False,          # Boolean [True or False] NOTE: Makes horizontal flip
        vertical_flip=False,            # Boolean [True or False] NOTE: Makes vertical flip
        fill_mode='constant',cval=0)    # All fill mode values [nearest, constant, reflect, wrap], cavl=outborder Color # Range int [0-255]
                                        # NOTE: nearest -> Stretch the boundaries' pixel values
                                        # NOTE: constant -> Outside boundaries will be defined color "cval=k"
                                        # NOTE: reflect -> Use reflected images to fill the boundaries
                                        # NOTE: wrap -> Paste the same image in all directions

# ---------> Read the image <--------#
x = cv2.imread('../../developmental_phase_images/H_manually_extracted.png') # Read image as grayscale, also the same code works with color images

# ---------> Reshape input image because datagen.flow() function demands 4 dimensional array <--------#
x = x.reshape((1, ) + x.shape)  #Array with shape (1, x , x , x)
i = 0

# ---------> Delete all previous images from directory <--------#
shutil.rmtree('../../developmental_phase_images/1.0_Data_augmentation_created_images') # Delete all directory with images
os.mkdir("../../developmental_phase_images/1.0_Data_augmentation_created_images") # Make empty directory with the same name

saveGeneratedImagesToTheDirectoryPath= "../../developmental_phase_images/1.0_Data_augmentation_created_images"

# ---------> Generate images <--------#
for batch in datagen.flow(x,                            # x-input image in array from
                          batch_size=16,                # batch_size - number of the image creations
                          save_to_dir=saveGeneratedImagesToTheDirectoryPath,      # save_to_dir - directory where you want to save generated images
                          save_prefix='aug',            # save_prefix - add wanted prefix to the
                          save_format='png'):           # save_format - defines format in which images will be saved

    # NOTE: This datagen.flow() function will be run infinitely, because of that we need something what will break it
    i += 1
    if i > 20:                                          # Define how many images you want to create
        break                                           # Break the function
