import cv2
import numpy as np


#---------> Load input image as grayscale and extract image dimensions <--------#
path= '1_images_for_read_show_and_stop_image_from_directory/H_more_than_W.PNG'  # Path of wanted input image
imgObjectSelected = cv2.imread(path)[:,:,0]               # Read image
cv2.imshow("imgObjectSelected",imgObjectSelected)         # Display image
print("imgObjectSelected shape:",imgObjectSelected.shape) # Print the image shape in console
heght = imgObjectSelected.shape[0]                        # Get input image height
width = imgObjectSelected.shape[1]                        # Get input image width

#---------> Resize input image to wanted dimension <--------#
if width > heght:                                        # Looking for the maximal dimension of input image
    maxSelectedObjectDimension = width
else:
    maxSelectedObjectDimension = heght

wantedMaxSelectedObjImageDimension = 28                  # Define maximal dimension after scaling
scalingRate = maxSelectedObjectDimension / wantedMaxSelectedObjImageDimension       # Get scale ratio
#print("Scalerate is:", scalingRate)                                                # Print the sale rate
finalHeight=int(heght/scalingRate )
finalWidth=int(width/scalingRate )
imgObjectSelectedResized = cv2.resize(imgObjectSelected, (finalWidth,finalHeight))       # Scale image
cv2.imshow("imgObjectSelectedResized",imgObjectSelectedResized)                          # Display resized image in console
print("imgObjectSelectedResized:",imgObjectSelectedResized.shape)                        # Print resized image in console
heght = imgObjectSelectedResized.shape[0]                                                # Get image height
width = imgObjectSelectedResized.shape[1]                                                # Get image width
# NOTE: Our CNN model is trained on the images with white color letter on black color background
# NOTE: Because of that we will invert color on our selected object image
# NOTE: If you don't want to invert color values comment three lines bellow
# NOTE: and make sure imgObjectSelectedResized stays everywhere instead imgObjectSelectedResizedClrInverted
imgObjectSelectedResizedClrInverted = np.invert(np.array([imgObjectSelectedResized]))    # Inverting image pixels value e.g: black color to white or reverse
print('imgObjectSelectedResizedClrInverted', imgObjectSelectedResizedClrInverted.shape)  # Print image shape
#print('imgObjectSelectedResizedClrInverted', imgObjectSelectedResizedClrInverted)       # Shows every image pixel value

if heght > width:
    makeImgToImgOffsetOnAxsisX=int((heght-width)/2)
    makeImgToImgOffsetOnAxsisY =0
else:
    makeImgToImgOffsetOnAxsisY = int((width - heght)/2)
    makeImgToImgOffsetOnAxsisX = 0


#---------> Adding black background in order to get 28x28 image which will be ready for input in CNN <--------#
blackBcgImg = cv2.imread("3_Backgound_black_28x28_image/Black_background.png")[:,:,0]   # Load image with one channel "grayscale"
cv2.imshow("blackBcgImg",blackBcgImg)                                                   # Display image
print("blackBcgImg:",blackBcgImg.shape)
# NOTE: Upper left corner of every image has coordinates 0,0
drawImgToImagStartCordinateX = makeImgToImgOffsetOnAxsisX                               # X coordinate where will start our inserting image
drawImgToImagStartCordinateY = makeImgToImgOffsetOnAxsisY                               # Y coordinate where will start our inserting image
blackBcgImg[drawImgToImagStartCordinateY : drawImgToImagStartCordinateY + heght , drawImgToImagStartCordinateX: drawImgToImagStartCordinateX + width] = imgObjectSelectedResizedClrInverted # Paste image in to another image
print("imgToImgCoordinates: ",drawImgToImagStartCordinateY, drawImgToImagStartCordinateY + heght,drawImgToImagStartCordinateX, drawImgToImagStartCordinateX + width)
cv2.imshow('imgToImg', blackBcgImg)
print("imgToImg",blackBcgImg.shape)

cv2.waitKey(0) # Pause the program until any key is pressed