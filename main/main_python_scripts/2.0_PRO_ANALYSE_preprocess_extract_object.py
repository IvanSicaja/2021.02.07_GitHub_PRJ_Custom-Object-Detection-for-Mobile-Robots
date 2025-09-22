import cv2
import numpy as np


# The function needed for the Tracking bars which, the function does nothing
def nothing(x):
    pass


# The function which add text and background rectangle on the image
def addTextWithBackgroundRectangleToImage(img, wantedText, startCoordinateX=0, startCoordinateY=0, textThickness=2, textScale=1, text_width_rectangle_offest=1.05, text_height_rectangle_offest=1.9, textStartOffest=5):

    (test_width, text_height), baseline = cv2.getTextSize(wantedText, cv2.FONT_HERSHEY_SIMPLEX, textScale, textThickness)  # Getting the text size in pixels because we want draw the background rectangle around the text, OUTPUT SHAPE: cv.getTextSize(text, fontFace, fontScale, textThickness) -> output=((189, 22), 10)
    # print('BB text size: ', test_width, text_height)
    cv2.rectangle(img, (startCoordinateX, startCoordinateY), (startCoordinateX + int(text_width_rectangle_offest * test_width),startCoordinateY + int(text_height_rectangle_offest * text_height)), (255, 255, 255), cv2.FILLED)  # Draw the filled color rectangle, NOTE: the drawing starts from the upper left corner and ends in the lower right corner
    cv2.putText(img, wantedText, (startCoordinateX + textStartOffest, startCoordinateY + text_height + textStartOffest),cv2.FONT_HERSHEY_SIMPLEX, textScale, (0, 0, 0), textThickness)                                            # Add the text to the OpenCV window

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
            objCor = len(detectedNumberOfEdgesOnContur)                                          # Store the number of edges on the contour in a variable
            x, y, w, h = cv2.boundingRect(detectedNumberOfEdgesOnContur)                         # Returns coordinates, width and height for drawing bounding box around contour, NOTE: the drawing starts from the upper left corner and ends in the lower right corner☺☺

            if objCor == 3:
                objectType = "Tri"  # Classificator related with the number od contour's edges
            elif objCor == 4:  # Classificator related with the number od contour's edges
                aspRatio = w / float(h)  # Square and rectangle classificator
                if aspRatio > 0.98 and aspRatio < 1.03:
                    objectType = "Square"
                else:
                    objectType = "Rectangle"
            elif objCor > 4:
                objectType = "Circles"
            else:
                objectType = "None"

            text_width_rectangle_offest = 1.1
            text_height_rectangle_offest = 1.7
            textScale = 0.5
            textThickness = 1
            putTextOffest = textThickness + 3
            boundingBoxRectangleThickness = 2

            wantedText = "Selection"
            startCoordinateX = x
            startCoordinateY = y

            cv2.rectangle(onImgDrawContours, (startCoordinateX, startCoordinateY),
                          (startCoordinateX + w, startCoordinateY + h), (0, 255, 0),
                          boundingBoxRectangleThickness)  # Draw the rectangle around the contour
            (test_width, text_height), baseline = cv2.getTextSize(wantedText, cv2.FONT_HERSHEY_SIMPLEX, textScale,
                                                                  textThickness)  # cv.getTextSize(	text, fontFace, fontScale, textThickness) -> output=((189, 22), 10)
            # print('BB text size: ', test_width, text_height)
            cv2.rectangle(onImgDrawContours, (startCoordinateX - boundingBoxRectangleThickness, startCoordinateY), (
            startCoordinateX + int(text_width_rectangle_offest * test_width),
            startCoordinateY + int(text_height_rectangle_offest * text_height)), (0, 255,), cv2.FILLED)
            cv2.putText(onImgDrawContours, wantedText,
                        (startCoordinateX + putTextOffest, startCoordinateY + text_height + putTextOffest),
                        cv2.FONT_HERSHEY_SIMPLEX, textScale, (0, 0, 0), textThickness)

            if w > h:
                extend_inital_object_selection_percent = int(0.1 * w)
            else:
                extend_inital_object_selection_percent = int(0.1 * h)

            text_width_rectangle_offest = 1.1
            text_height_rectangle_offest = 1.7
            textScale = 0.5
            textThickness = 1
            putTextOffest = textThickness + 3
            boundingBoxRectangleThickness = 2

            wantedText = "Extended selection"
            startCoordinateX = x - int(extend_inital_object_selection_percent / 2)
            startCoordinateY = y + h + int(extend_inital_object_selection_percent / 2)

            cv2.rectangle(onImgDrawContours, (
            x - int(extend_inital_object_selection_percent / 2), y - int(extend_inital_object_selection_percent / 2)), (
                          x + w + int(extend_inital_object_selection_percent / 2),
                          y + h + int(extend_inital_object_selection_percent / 2)), (0, 255, 0),
                          boundingBoxRectangleThickness)  # Draw the rectangle around the contour
            (test_width, text_height), baseline = cv2.getTextSize(wantedText, cv2.FONT_HERSHEY_SIMPLEX, textScale,
                                                                  textThickness)  # cv.getTextSize(	text, fontFace, fontScale, textThickness) -> output=((189, 22), 10)
            # print('BB text size: ', test_width, text_height)
            cv2.rectangle(onImgDrawContours, (startCoordinateX - boundingBoxRectangleThickness, startCoordinateY), (
            startCoordinateX + int(text_width_rectangle_offest * test_width),
            startCoordinateY + int(text_height_rectangle_offest * text_height)), (0, 255,), cv2.FILLED)
            cv2.putText(onImgDrawContours, wantedText,
                        (startCoordinateX + putTextOffest, startCoordinateY + text_height + putTextOffest),
                        cv2.FONT_HERSHEY_SIMPLEX, textScale, (0, 0, 0), textThickness)

            boundingBoxStart_EndCoordinates = [x - int(extend_inital_object_selection_percent / 2),
                                               y - int(extend_inital_object_selection_percent / 2),
                                               x + w + int(extend_inital_object_selection_percent / 2),
                                               y + h + int(extend_inital_object_selection_percent / 2)]

            imgObjectSelected = fromImgExtractObject[
                                boundingBoxStart_EndCoordinates[1]:boundingBoxStart_EndCoordinates[3],
                                boundingBoxStart_EndCoordinates[0]:boundingBoxStart_EndCoordinates[
                                    2]]  # First height range, then width range
            print("imgObjectSelected:", imgObjectSelected)
            print("Number of iteration:", iterationCounter)

            cv2.imshow("Image of object", imgObjectSelected)


        else:
            print("Contour is smaller then defined area")


cv2.namedWindow('Ultimate Computer Vision Script')  # Create OpenCV window
cv2.createTrackbar('LCanny', 'Ultimate Computer Vision Script', 0, 255, nothing)  # Create trackbar
cv2.createTrackbar('HCanny', 'Ultimate Computer Vision Script', 0, 255, nothing)  # Create trackbar
cv2.setTrackbarPos('LCanny', 'Ultimate Computer Vision Script', 30)  # Set default trackbars values
cv2.setTrackbarPos('HCanny', 'Ultimate Computer Vision Script', 90)  # Set default trackbars values
iterationCounter = 0  # Count the number of calling getContours() function

while True:  # Loop our program

    # NOTE: ".." in the path means go to the parent's directory
    path = '../../developmental_phase_images/H_S_U.png'  # Wanted image
    img = cv2.imread(path)  # Read the wanted image
    # print("Input image shape:", img.shape)
    imgContour = img.copy()  # Copy the wanted image, on this one we will draw the bounding-boxes

    LCanny = cv2.getTrackbarPos('LCanny', 'Ultimate Computer Vision Script')  # Read value from trackbar
    HCanny = cv2.getTrackbarPos('HCanny', 'Ultimate Computer Vision Script')  # Read value from trackbar

    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # Convert input image in to Grayscale = 3 image channels is replaced with 1 chanel
    # NOTE: Contrast layer is not needed because we have amazing Threshold layer
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)  # Remove noise from Grayscale image
    imgThreshold = cv2.threshold(imgBlur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # Count and return optimal Threshold value for the input image
    imgCanny = cv2.Canny(imgThreshold, LCanny, HCanny)  # Extract edges from image
    getContours(imgCanny, imgContour, imgThreshold, iterationCounter)  # Call the function for the contours processing
    iterationCounter = iterationCounter + 1  # Increase the number of calling getContours() function
    kernel = np.ones((5, 5))  # Define matrix size of Kernel
    imgDila = cv2.dilate(imgCanny, kernel, iterations=2)  # Expand size of found contours from the Canny image
    imgErode = cv2.erode(imgDila, kernel, iterations=1)  # Reduce size of found contours from the Dilated image
    imgBlank = np.zeros_like(img)  # Create black image, USAGE-> fill the stack image free place

    addTextWithBackgroundRectangleToImage(img, "Input image")  # Add text and background rectangle on the image
    addTextWithBackgroundRectangleToImage(imgGray, "Grayscale image")
    addTextWithBackgroundRectangleToImage(imgBlur, "Blur image")
    addTextWithBackgroundRectangleToImage(imgThreshold, "Treshold image")
    addTextWithBackgroundRectangleToImage(imgCanny, "Canny image")
    addTextWithBackgroundRectangleToImage(imgDila, "Dilatated image")
    addTextWithBackgroundRectangleToImage(imgErode, "Erode image")
    addTextWithBackgroundRectangleToImage(imgContour, "Contours")
    addTextWithBackgroundRectangleToImage(imgBlank, "Noting")
    addTextWithBackgroundRectangleToImage(imgBlank, "Noting")

    imgStack = stackImages(0.4, ([img, imgGray, imgBlur], [imgThreshold, imgCanny, imgDila],
                                 [imgErode, imgContour, imgBlank]))  # Add all the images on the one window
    cv2.imshow("window", imgStack)  # Show created OpenCV window

    if cv2.waitKey(1) == 27:  # Listen if the "ESC" key is pressed
        cv2.destroyAllWindows()  # Close all OpenCV windows
        break  # Beak the while loop = the program is finished
