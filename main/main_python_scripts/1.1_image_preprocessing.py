import cv2
import numpy as np
import matplotlib.pyplot as plt

def nothing(x): # The function for trecking bars
	pass

def addTextWithBackroundRectangleToImage(img,wantedText,startCoordinateX=0, startCoordinateY=0,textThickness=2,textScale=1,text_width_rectangle_offest = 1.05,text_height_rectangle_offest = 1.9,textStartOffest = 5):

    (test_width, text_height), baseline = cv2.getTextSize(wantedText, cv2.FONT_HERSHEY_SIMPLEX, textScale, textThickness)  # cv.getTextSize(	text, fontFace, fontScale, textThickness) -> output=((189, 22), 10)
    # print('BB text size: ', test_width, text_height)
    cv2.rectangle(img, (startCoordinateX, startCoordinateY), (startCoordinateX + int(text_width_rectangle_offest * test_width),startCoordinateY + int(text_height_rectangle_offest * text_height)), (255, 255, 255), cv2.FILLED)
    cv2.putText(img, wantedText, (startCoordinateX + textStartOffest, startCoordinateY + text_height + textStartOffest),cv2.FONT_HERSHEY_SIMPLEX, textScale, (0, 0, 0), textThickness)

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

cv2.namedWindow('Ultimate Computer Vision Script')
cv2.createTrackbar('LVal', 'Ultimate Computer Vision Script', 0, 255, nothing)  # Create trackbar 1
cv2.createTrackbar('UVal', 'Ultimate Computer Vision Script', 0, 255, nothing)  # Create trackbar 1

cv2.setTrackbarPos( 'LVal', 'Ultimate Computer Vision Script', 50)  # Set default trackbars values
cv2.setTrackbarPos( 'UVal', 'Ultimate Computer Vision Script', 50)  # Set default trackbars values

path= '1.0_character_images_for_bounding_boxes/H.png'  # Wanted image
img = cv2.imread(path)                                  # Read the wanted image
#print("Input image shape:", img.shape)

while True:

    LVal = cv2.getTrackbarPos('LVal', 'Ultimate Computer Vision Script')  # Read value from trackbar
    UVal = cv2.getTrackbarPos('UVal', 'Ultimate Computer Vision Script')  # Read value from trackbar

    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #Contrast
    imgBlur =cv2.GaussianBlur(imgGray,(5,5),1)
    #imgTreshold=cv2.threshold(imgBlur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    imgCanny = cv2.Canny(imgBlur,LVal,UVal)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel,iterations=2)
    imgErode = cv2.erode(imgDial,kernel,iterations=1)
    imgBlank = np.zeros_like(img)                   # Fill the stack image free place



    addTextWithBackroundRectangleToImage(img,"Input image")
    addTextWithBackroundRectangleToImage(imgGray,"Grayscale image")
    addTextWithBackroundRectangleToImage(imgBlur,"Blur image")
    addTextWithBackroundRectangleToImage(imgCanny,"Canny image")
    addTextWithBackroundRectangleToImage(imgDial,"Dilatated image")
    addTextWithBackroundRectangleToImage(imgErode,"Erode image")
    addTextWithBackroundRectangleToImage(imgBlank,"Nothing")



    imgStack = stackImages(0.5,([img,imgGray,imgBlur],[imgCanny,imgDial,imgErode]))
    cv2.imshow("Ultimate Computer Vision Script", imgStack)

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        break



'''
titles = ['Input image', 'Grayscale image', 'Blur image', 'Canny image', 'Erode image', 'Erode image']
images = [img,imgGray,imgBlur,imgCanny,imgDial,imgErode]

scaleFactorForPlot=1
imagesHeightForPloting= scaleFactorForPlot*img.shape[0]
imagesWidhtForPloting= scaleFactorForPlot*img.shape[1]

for i in range(6):

    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()
'''
