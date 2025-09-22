import cv2
import numpy as np


# The function needed for the Tracking bars which, the function does nothing
def nothing(x):
    pass

# The function which add text and background rectangle on the image
def addTextWithBackgroundRectangleToImage(img, wantedText, startCoordinateX=0, startCoordinateY=0, textThickness=2, textBackgroundColor= (255, 255, 255), textScale=1, text_width_rectangle_offest=1.05, text_height_rectangle_offest=1.9, textStartOffest=5):

    (test_width, text_height), baseline = cv2.getTextSize(wantedText, cv2.FONT_HERSHEY_SIMPLEX, textScale, textThickness)  # Getting the text size in pixels because we want draw the background rectangle around the text, OUTPUT SHAPE: cv.getTextSize(text, fontFace, fontScale, textThickness) -> output=((189, 22), 10)
    # print('BB text size: ', test_width, text_height)
    cv2.rectangle(img, (startCoordinateX, startCoordinateY), (startCoordinateX + int(text_width_rectangle_offest * test_width),startCoordinateY + int(text_height_rectangle_offest * text_height)), textBackgroundColor, cv2.FILLED)  # Draw the filled color rectangle, NOTE: the drawing starts from the upper left corner and ends in the lower right corner
    cv2.putText(img, wantedText, (startCoordinateX + textStartOffest, startCoordinateY + text_height + textStartOffest),cv2.FONT_HERSHEY_SIMPLEX, textScale, (0, 0, 0), textThickness)                                            # Add the text to the OpenCV window

# The function which add text and background rectangle on left bellow corner of the bouding box
def addTextWithBackgroundRectangleToImageBellowBoundingBox(img, wantedText, startCoordinateX=0, startCoordinateY=0,boundingBoxThickness=2 ,textThickness=2, textBackgroundColor= (255, 255, 255), textScale=1, text_width_rectangle_offest=1.05, text_height_rectangle_offest=1.9, textStartOffest=5):

    (test_width, text_height), baseline = cv2.getTextSize(wantedText, cv2.FONT_HERSHEY_SIMPLEX, textScale, textThickness)  # Getting the text size in pixels because we want draw the background rectangle around the text, OUTPUT SHAPE: cv.getTextSize(text, fontFace, fontScale, textThickness) -> output=((189, 22), 10)
    # print('BB text size: ', test_width, text_height)
    cv2.rectangle(img, (startCoordinateX-int(boundingBoxThickness/2), startCoordinateY), (startCoordinateX + int(text_width_rectangle_offest * test_width)-boundingBoxThickness,startCoordinateY + int(text_height_rectangle_offest * text_height)), textBackgroundColor, cv2.FILLED)  # Draw the filled color rectangle, NOTE: the drawing starts from the upper left corner and ends in the lower right corner
    cv2.putText(img, wantedText, (startCoordinateX + textStartOffest-boundingBoxThickness, startCoordinateY + text_height + textStartOffest),cv2.FONT_HERSHEY_SIMPLEX, textScale, (0, 0, 0), textThickness)  # Add the text to the OpenCV window


# The function which add images one next or bellow another on the same OpenCV window
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def getContours(img, onImgDrawContours, fromImgExtractObject, iterationCounter):

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)         # Finding all contours on the image
    for cnt in contours:                                                                         # Individually access to every contour
        area = cv2.contourArea(cnt)                                                              # Calculate area of every image
        print("Contours area: ", area)                                                           # Print the size in pixels of contour

        if area > 17:
            cv2.drawContours(onImgDrawContours, cnt, -1, (255, 0, 0),3)                          # Draw contours "cnt" on the image "onImgDrawContours"
            arcLengthValue = cv2.arcLength(cnt, True)                                            # Calculate the acr length of the contour
            # print("arcLengthValue: ",arcLengthValue)                                           # Print the acr length of the contours
            detectedNumberOfEdgesOnContur = cv2.approxPolyDP(cnt, 0.02 * arcLengthValue, True)   # Looking for number of edges on the contour
            print("Number of edges for selected contour: ", len(detectedNumberOfEdgesOnContur))  # Print the number of edges on the contour
            numberOfEdgesOnContur = len(detectedNumberOfEdgesOnContur)                                          # Store the number of edges on the contour in a variable
            x, y, w, h = cv2.boundingRect(detectedNumberOfEdgesOnContur)                         # Returns coordinates, width and height for drawing bounding box around contour, NOTE: the drawing starts from the upper left corner and ends in the lower right corner☺☺

            # NOTE: This "if,elif,else" pat bellow you can use for the: triangle, rectangle and circle part
            # if numberOfEdgesOnContur == 3:                                                     # Classificator related with the number od contour's edges
            #     objectType = "Triangle"                                                        # Triangle classificator
            # elif numberOfEdgesOnContur == 4:                                                   # Classificator related with the number od contour's edges
            #     aspRatio = w / float(h)                                                        # Square and rectangle classificator
            #     if aspRatio > 0.98 and aspRatio < 1.03:                                        # Difference between the square and rectangle
            #         objectType = "Square"
            #     else:
            #         objectType = "Rectangle"
            # elif numberOfEdgesOnContur > 4:                                                    # Circle and rectangle classificator
            #     objectType = "Circles"
            # else:
            #     objectType = "None"

            detectedObjectStartCoordinateX = x                                                   # Left X coordinate of the detected object
            detectedObjectStartCoordinateY = y                                                   # Upper Y coordinate of the detected object
            cv2.rectangle(onImgDrawContours, (detectedObjectStartCoordinateX, detectedObjectStartCoordinateY), (detectedObjectStartCoordinateX + w, detectedObjectStartCoordinateY + h), (0, 255, 0), 2)  # Draw the rectangle around the contour i.e. around detected object= bounding box
            addTextWithBackgroundRectangleToImage(onImgDrawContours,"Selection", detectedObjectStartCoordinateX, detectedObjectStartCoordinateY, 2, (0, 255, 00), 0.7)                                    # Add text with background in upper left corner on bounding box

            boundingBoxExtensionPercent=0.1                                                      # Define extension percent, because we want to be sure that the entire object in inner the bounding box
            if w > h:                                                                            # Looking for the larger object dimension
                extendInitialObjectSelection = int(boundingBoxExtensionPercent * w)              # Calculate the number of pixel needed for the extension
            else:
                extendInitialObjectSelection = int(boundingBoxExtensionPercent * h)              # Calculate the number of pixel needed for the extension

            textBelowBondingBoxStartCoordinateX = x - int(extendInitialObjectSelection / 2)      # Starting coordinates for writing the text bellow the bounding box
            textBelowBondingBoxStartCoordinateY = y + h + int(extendInitialObjectSelection / 2)  # Starting coordinates for writing the text bellow the bounding box

                                                                                                 # Start and end coordinates for the extended bounding box
            boundingBoxStart_EndCoordinates = [x - int(extendInitialObjectSelection / 2),        # Coordinate X upper left corner of the extended bounding box
                                               y - int(extendInitialObjectSelection / 2),        # Coordinate Y upper left corner of the extended bounding box
                                               x + w + int(extendInitialObjectSelection / 2),    # Coordinate X lower right corner of the extended bounding box
                                               y + h + int(extendInitialObjectSelection / 2)]    # Coordinate Y lower right corner of the extended bounding box

            cv2.rectangle(onImgDrawContours, (x - int(extendInitialObjectSelection / 2), y - int(extendInitialObjectSelection / 2)), (x + w + int(extendInitialObjectSelection / 2),y + h + int(extendInitialObjectSelection / 2)), (0, 255, 0),2)  # Draw the extended bounding box around the contour
            addTextWithBackgroundRectangleToImageBellowBoundingBox(onImgDrawContours,"Extension", textBelowBondingBoxStartCoordinateX, textBelowBondingBoxStartCoordinateY,2,2, (0, 255, 00), 0.7)                                                  # Add text with background in lower left corner on bounding box
            # NOTE: If we want to extract part of the image, first goes height pixel interval, then wanted weight pixel interval
            imgObjectSelected = fromImgExtractObject[                                                   # Crop space inner extended bounding box
                                boundingBoxStart_EndCoordinates[1]:boundingBoxStart_EndCoordinates[3],  # X inner rage of extended bounding box
                                boundingBoxStart_EndCoordinates[0]:boundingBoxStart_EndCoordinates[2]]

            print("imgObjectSelected:", imgObjectSelected)                                              # Print extracted object values in the console
            print("Number of iteration:", iterationCounter)                                             # Print number of calls of getContours() function
            cv2.imshow("Image of object", imgObjectSelected)                                            # Show extracted object from extended bounding box in cv2 window
            # NOTE: If you want to show every detected image step by step uncomment the lines bellow, for the next image you need to press any key, for the exit from the program press "ESC" key two times
            # k=cv2.waitKey(0)                                                                          # Stops the program until any key is pressed
            # if k == 27:                                                                               # Listen if the "ESC" key is pressed
            #     cv2.destroyAllWindows()                                                               # Close all OpenCV windows
            #     break

        else:                                                                                           # To this if all contours are smaller of defined contours area
            print("Contour is smaller then defined area")


cv2.namedWindow('Ultimate Computer Vision Script')                                                      # Create OpenCV window
cv2.createTrackbar('LCanny', 'Ultimate Computer Vision Script', 0, 255, nothing)                        # Create trackbar
cv2.createTrackbar('HCanny', 'Ultimate Computer Vision Script', 0, 255, nothing)                        # Create trackbar
cv2.setTrackbarPos('LCanny', 'Ultimate Computer Vision Script', 30)                                     # Set default trackbars values
cv2.setTrackbarPos('HCanny', 'Ultimate Computer Vision Script', 90)                                     # Set default trackbars values
iterationCounter = 0                                                                                    # Count the number of calling getContours() function

while True:                                                                                             # Loop our program

    # NOTE: ".." in the path means go to the parent's directory
    path = '../../developmental_phase_images/H_S_U.png'  # Wanted image
    img = cv2.imread(path)                                                                              # Read the wanted image
    # print("Input image shape:", img.shape)
    imgContour = img.copy()                                                                             # Copy the wanted image, on this one we will draw the bounding-boxes

    LCanny = cv2.getTrackbarPos('LCanny', 'Ultimate Computer Vision Script')                            # Read value from trackbar
    HCanny = cv2.getTrackbarPos('HCanny', 'Ultimate Computer Vision Script')                            # Read value from trackbar

    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                                                      # Convert input image in to Grayscale = 3 image channels is replaced with 1 chanel
    # NOTE: Contrast layer is not needed because we have amazing Threshold layer
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)                                                      # Remove noise from Grayscale image
    imgThreshold = cv2.threshold(imgBlur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]               # Count and return optimal Threshold value for the input image
    imgCanny = cv2.Canny(imgThreshold, LCanny, HCanny)                                                  # Extract edges from image
    getContours(imgCanny, imgContour, imgThreshold, iterationCounter)                                   # Call the function for the contours processing
    iterationCounter = iterationCounter + 1                                                             # Increase the number of calling getContours() function
    kernel = np.ones((5, 5))              # Define matrix size of Kernel
    imgDila = cv2.dilate(imgCanny, kernel, iterations=2)                                                # Expand size of found contours from the Canny image
    imgErode = cv2.erode(imgDila, kernel, iterations=1)                                                 # Reduce size of found contours from the Dilated image
    imgBlank = np.zeros_like(img)                                                                       # Create black image, USAGE-> fill the stack image free place

    addTextWithBackgroundRectangleToImage(img, "Input image")                                           # Add text and background rectangle on the image
    addTextWithBackgroundRectangleToImage(imgGray, "Grayscale image")
    addTextWithBackgroundRectangleToImage(imgBlur, "Blur image")
    addTextWithBackgroundRectangleToImage(imgThreshold, "Treshold image")
    addTextWithBackgroundRectangleToImage(imgCanny, "Canny image")
    addTextWithBackgroundRectangleToImage(imgDila, "Dilatated image")
    addTextWithBackgroundRectangleToImage(imgErode, "Erode image")
    addTextWithBackgroundRectangleToImage(imgContour, "Contours")
    addTextWithBackgroundRectangleToImage(imgBlank, "Noting")
    addTextWithBackgroundRectangleToImage(imgBlank, "Noting")

    imgStack = stackImages(0.4, ([img, imgGray, imgBlur],                                              # Add all the images on the one window
                                 [imgThreshold, imgCanny, imgDila],
                                 [imgErode, imgContour, imgBlank]))
    cv2.imshow("Ultimate Computer Vision Script", imgStack)                                            # Show created OpenCV window

    if cv2.waitKey(0) == 27:                                                                           # Listen if the "ESC" key is pressed
        cv2.destroyAllWindows()                                                                        # Close all OpenCV windows
        break                                                                                          # Beak the while loop = the program is finished
