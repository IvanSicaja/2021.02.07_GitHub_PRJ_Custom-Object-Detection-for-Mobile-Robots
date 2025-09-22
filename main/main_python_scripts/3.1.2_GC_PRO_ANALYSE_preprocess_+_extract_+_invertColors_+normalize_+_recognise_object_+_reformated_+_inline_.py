import cv2
import numpy as np
from keras.models import load_model  # Used for load trained CNN model


# The function needed for the Tracking bars which, the function does nothing
def nothing(x):
    pass


# The function which add text and background rectangle on the image
def addTextWithBackgroundRectangleToImage(img, wantedText, startCoordinateX=0, startCoordinateY=0, textThickness=2, textBackgroundColor=(255, 255, 255), textScale=1, text_width_rectangle_offest=1.05, text_height_rectangle_offest=1.9, textStartOffest=5):
    (test_width, text_height), baseline = cv2.getTextSize(wantedText, cv2.FONT_HERSHEY_SIMPLEX, textScale, textThickness)  # Getting the text size in pixels because we want draw the background rectangle around the text, OUTPUT SHAPE: cv.getTextSize(text, fontFace, fontScale, textThickness) -> output=((189, 22), 10)
    # print('BB text size: ', test_width, text_height)
    cv2.rectangle(img, (startCoordinateX, startCoordinateY), (startCoordinateX + int(text_width_rectangle_offest * test_width), startCoordinateY + int(text_height_rectangle_offest * text_height)), textBackgroundColor, cv2.FILLED)  # Draw the filled color rectangle, NOTE: the drawing starts from the upper left corner and ends in the lower right corner
    cv2.putText(img, wantedText, (startCoordinateX + textStartOffest, startCoordinateY + text_height + textStartOffest), cv2.FONT_HERSHEY_SIMPLEX, textScale, (0, 0, 0), textThickness)  # Add the text to the OpenCV window


# The function which add text and background rectangle on left bellow corner of the bouding box
def addTextWithBackgroundRectangleToImageBellowBoundingBox(img, wantedText, startCoordinateX=0, startCoordinateY=0, boundingBoxThickness=2, textThickness=2, textBackgroundColor=(255, 255, 255), textScale=1, text_width_rectangle_offest=1.05, text_height_rectangle_offest=1.9, textStartOffest=5):
    (test_width, text_height), baseline = cv2.getTextSize(wantedText, cv2.FONT_HERSHEY_SIMPLEX, textScale, textThickness)  # Getting the text size in pixels because we want draw the background rectangle around the text, OUTPUT SHAPE: cv.getTextSize(text, fontFace, fontScale, textThickness) -> output=((189, 22), 10)
    # print('BB text size: ', test_width, text_height)
    cv2.rectangle(img, (startCoordinateX - int(boundingBoxThickness / 2), startCoordinateY), (startCoordinateX + int(text_width_rectangle_offest * test_width) - boundingBoxThickness, startCoordinateY + int(text_height_rectangle_offest * text_height)), textBackgroundColor, cv2.FILLED)  # Draw the filled color rectangle, NOTE: the drawing starts from the upper left corner and ends in the lower right corner
    cv2.putText(img, wantedText, (startCoordinateX + textStartOffest - boundingBoxThickness, startCoordinateY + text_height + textStartOffest), cv2.FONT_HERSHEY_SIMPLEX, textScale, (0, 0, 0), textThickness)  # Add the text to the OpenCV window


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
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
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


# ----------> Create all needed trackbars <----------#
cv2.namedWindow('Sliders')  # Create OpenCV window
cv2.createTrackbar('LCanny', 'Sliders', 0, 255, nothing)  # Create trackbar
cv2.createTrackbar('HCanny', 'Sliders', 0, 255, nothing)  # Create trackbar
cv2.setTrackbarPos('LCanny', 'Sliders', 30)  # Set default trackbars values
cv2.setTrackbarPos('HCanny', 'Sliders', 90)  # Set default trackbars values
cv2.createTrackbar('scalePrewImg', 'Sliders', 1, 15, nothing)  # Create trackbar
cv2.setTrackbarPos('scalePrewImg', 'Sliders', 4)

iterationCounter = 0  # Count the number of calling getContours() function

# ----------> Load trained model with which we will recognise character from the object <----------#
model = load_model('../../4.0_Good_saved_CNN_models/2.0_A-Z_0-9/Model_Batchsize=500__Epoch=60__Accuracy=77.5.h5')  # Load trained wanted model
model.summary()  # Load the model summary, can be useful to see CNN layers structure

# ----------> Main program loop <----------#
while True:  # Loop our program
    # ----------> Get all trackbars values and save it in the variable <----------#
    LCanny = cv2.getTrackbarPos('LCanny', 'Sliders')  # Read value from trackbar
    HCanny = cv2.getTrackbarPos('HCanny', 'Sliders')  # Read value from trackbar
    scalePreviewImagesValue = cv2.getTrackbarPos('scalePrewImg', 'Sliders')

    # ----------> Define image/video source from witch we went recognise object <----------#
    # NOTE: ".." in the path means go to the parent's directory
    path = '../../developmental_phase_images/H_S_U.png'  # Wanted image
    img = cv2.imread(path)  # Read the wanted image
    # print("Input image shape:", img.shape)
    onImgDrawContours = img.copy()

    # ----------> Do image preprocessing <----------#
    # Copy the wanted image, on this one we will draw the bounding-boxes
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert input image in to Grayscale = 3 image channels is replaced with 1 chanel
    # NOTE: Contrast layer is not needed because we have amazing Threshold layer
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)  # Remove noise from Grayscale image
    imgThreshold = cv2.threshold(imgBlur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # Count and return optimal Threshold value for the input image
    imgCanny = cv2.Canny(imgThreshold, LCanny, HCanny)  # Extract edges from image
    iterationCounter = iterationCounter + 1  # Increase the number of calling getContours() function
    kernel = np.ones((5, 5))  # Define matrix size of Kernel
    imgDila = cv2.dilate(imgCanny, kernel, iterations=2)  # Expand size of found contours from the Canny image
    imgErode = cv2.erode(imgDila, kernel, iterations=1)  # Reduce size of found contours from the Dilated image
    imgBlank = np.zeros_like(img)  # Create black image, USAGE-> fill the stack image free place

    # ----------> Find and extract object from the image <----------#
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # Finding all contours on the image
    for cnt in contours:  # Individually access to every contour
        area = cv2.contourArea(cnt)  # Calculate area of every image
        print("Contours area: ", str(area) + ' px')  # Print the size in pixels of contour

        if area > 17:
            cv2.drawContours(onImgDrawContours, cnt, -1, (255, 0, 0), 3)  # Draw contours "cnt" on the image "onImgDrawContours"
            arcLengthValue = cv2.arcLength(cnt, True)  # Calculate the acr length of the contour
            # print("arcLengthValue: ",arcLengthValue)                                           # Print the acr length of the contours
            detectedNumberOfEdgesOnContur = cv2.approxPolyDP(cnt, 0.02 * arcLengthValue, True)  # Looking for number of edges on the contour
            print("Number of edges for selected contour: ", len(detectedNumberOfEdgesOnContur))  # Print the number of edges on the contour
            numberOfEdgesOnContur = len(detectedNumberOfEdgesOnContur)  # Store the number of edges on the contour in a variable
            x, y, w, h = cv2.boundingRect(detectedNumberOfEdgesOnContur)  # Returns coordinates, width and height for drawing bounding box around contour, NOTE: the drawing starts from the upper left corner and ends in the lower right corner☺☺

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

            detectedObjectStartCoordinateX = x  # Left X coordinate of the detected object
            detectedObjectStartCoordinateY = y  # Upper Y coordinate of the detected object
            cv2.rectangle(onImgDrawContours, (detectedObjectStartCoordinateX, detectedObjectStartCoordinateY), (detectedObjectStartCoordinateX + w, detectedObjectStartCoordinateY + h), (0, 255, 0), 2)  # Draw the rectangle around the contour i.e. around detected object= bounding box
            addTextWithBackgroundRectangleToImage(onImgDrawContours, "Selection", detectedObjectStartCoordinateX, detectedObjectStartCoordinateY, 2, (0, 255, 00), 0.7)  # Add text with background in upper left corner on bounding box

            # ----------> Extend selected object selection <----------#
            boundingBoxExtensionPercent = 0.1  # Define extension percent, because we want to be sure that the entire object in inner the bounding box
            if w > h:  # Looking for the larger object dimension
                extendInitialObjectSelection = int(boundingBoxExtensionPercent * w)  # Calculate the number of pixel needed for the extension
            else:
                extendInitialObjectSelection = int(boundingBoxExtensionPercent * h)  # Calculate the number of pixel needed for the extension

            textBelowBondingBoxStartCoordinateX = x - int(extendInitialObjectSelection / 2)  # Starting coordinates for writing the text bellow the bounding box
            textBelowBondingBoxStartCoordinateY = y + h + int(extendInitialObjectSelection / 2)  # Starting coordinates for writing the text bellow the bounding box

            # Start and end coordinates for the extended bounding box
            boundingBoxStart_EndCoordinates = [x - int(extendInitialObjectSelection / 2),  # Coordinate X upper left corner of the extended bounding box
                                               y - int(extendInitialObjectSelection / 2),  # Coordinate Y upper left corner of the extended bounding box
                                               x + w + int(extendInitialObjectSelection / 2),  # Coordinate X lower right corner of the extended bounding box
                                               y + h + int(extendInitialObjectSelection / 2)]  # Coordinate Y lower right corner of the extended bounding box

            cv2.rectangle(onImgDrawContours, (x - int(extendInitialObjectSelection / 2), y - int(extendInitialObjectSelection / 2)), (x + w + int(extendInitialObjectSelection / 2), y + h + int(extendInitialObjectSelection / 2)), (0, 255, 0), 2)  # Draw the extended bounding box around the contour
            addTextWithBackgroundRectangleToImageBellowBoundingBox(onImgDrawContours, "Extension", textBelowBondingBoxStartCoordinateX, textBelowBondingBoxStartCoordinateY, 2, 2, (0, 255, 00), 0.7)  # Add text with background in lower left corner on bounding box

            # ----------> Extract object from extended selection <----------#
            # NOTE: If we want to extract part of the image, first goes height pixel interval, then wanted weight pixel interval
            imgObjectSelected = imgThreshold[  # Crop space inner extended bounding box
                                boundingBoxStart_EndCoordinates[1]:boundingBoxStart_EndCoordinates[3],  # X inner rage of extended bounding box
                                boundingBoxStart_EndCoordinates[0]:boundingBoxStart_EndCoordinates[2]]

            # print("imgObjectSelected:", imgObjectSelected)                                             # Print extracted object values in the console
            print("Number of iteration:", iterationCounter)  # Print number of calls of getContours() function
            cv2.imshow("Image of object", imgObjectSelected)  # Show extracted object from extended bounding box in cv2 window
            print("Extracted object shape: ", imgObjectSelected.shape)

            # TODO: Chech is height and width bellow uesd
            # heght = imgObjectSelected.shape[0]
            # width = imgObjectSelected.shape[1]

            cv2.imshow("imgObjectSelected", imgObjectSelected)  # Display image
            print("imgObjectSelected shape:", imgObjectSelected.shape)  # Print the image shape in console
            heght = imgObjectSelected.shape[0]  # Get input image height
            width = imgObjectSelected.shape[1]  # Get input image width

            # TODO: Maybe make a function of this block
            # ----------> Resize input image to wanted dimension <---------#
            if width > heght:  # Looking for the maximal dimension of input image
                maxSelectedObjectDimension = width
            else:
                maxSelectedObjectDimension = heght

            wantedMaxSelectedObjImageDimension = 28  # Define maximal dimension after scaling
            scalingRate = maxSelectedObjectDimension / wantedMaxSelectedObjImageDimension  # Get scale ratio
            # print("Scale-rate is:", scalingRate)                                                     # Print the sale rate
            finalHeight = int(heght / scalingRate)
            finalWidth = int(width / scalingRate)
            imgObjectSelectedResized = cv2.resize(imgObjectSelected, (finalWidth, finalHeight))  # Scale image
            cv2.imshow("imgObjectSelectedResized", imgObjectSelectedResized)  # Display resized image in console
            print("imgObjectSelectedResized:", imgObjectSelectedResized.shape)  # Print resized image in console
            heght = imgObjectSelectedResized.shape[0]  # Get image height
            width = imgObjectSelectedResized.shape[1]  # Get image width
            # NOTE: Our CNN model is trained on the images with white color letter on black color background
            # NOTE: Because of that we will invert color on our selected object image
            # NOTE: If you don't want to invert color values comment three lines bellow
            # NOTE: and make sure imgObjectSelectedResized stays everywhere instead imgObjectSelectedResizedClrInverted
            # NOTE: np.invert is giving batch-size dimension to image. e.g (28,28)->(1,28,28)
            imgObjectSelectedResizedClrInverted = np.invert(np.array([imgObjectSelectedResized]))  # Inverting image pixels value e.g: black color to white or reverse
            print('imgObjectSelectedResizedClrInverted', imgObjectSelectedResizedClrInverted.shape)  # Print image shape
            # print('imgObjectSelectedResizedClrInverted', imgObjectSelectedResizedClrInverted)       # Shows every image pixel value

            # ----------> Center the image on the black background image <----------#
            if heght > width:
                makeImgToImgOffsetOnAxsisX = int((heght - width) / 2)
                makeImgToImgOffsetOnAxsisY = 0
            else:
                makeImgToImgOffsetOnAxsisY = int((width - heght) / 2)
                makeImgToImgOffsetOnAxsisX = 0

            # ---------> Adding black background in order to get 28x28 image which will be ready for input in CNN <--------#
            blackBcgImg = cv2.imread("../../developmental_phase_images/Black_background.png")[:, :, 0]  # Load image with one channel "grayscale"
            cv2.imshow("blackBcgImg", blackBcgImg)  # Display image
            print("blackBcgImg:", blackBcgImg.shape)
            # NOTE: Upper left corner of every image has coordinates 0,0
            drawImgToImagStartCordinateX = makeImgToImgOffsetOnAxsisX  # X coordinate where will start our inserting image
            drawImgToImagStartCordinateY = makeImgToImgOffsetOnAxsisY  # Y coordinate where will start our inserting image
            blackBcgImg[drawImgToImagStartCordinateY: drawImgToImagStartCordinateY + heght, drawImgToImagStartCordinateX: drawImgToImagStartCordinateX + width] = imgObjectSelectedResizedClrInverted  # Paste image in to another image
            print("imgToImgCoordinates: ", drawImgToImagStartCordinateY, drawImgToImagStartCordinateY + heght, drawImgToImagStartCordinateX, drawImgToImagStartCordinateX + width)
            cv2.imshow('imgToImg', blackBcgImg)
            print("imgToImg :", blackBcgImg.shape)
            # NOTE: Shape needed for CNN input is (1,28,28)
            imgGoToCnn = np.reshape(blackBcgImg, (1, 28, 28, 1))
            print("imgGoToCnn: ", imgGoToCnn.shape)
            # NOTE: Needed values normalisation before the image goes in CNN
            imgGoToCnn = imgGoToCnn / 255.0

            # ----------> Send extracted object to the trained CNN model and predict object value <----------#
            pred = model.predict(imgGoToCnn)
            print("Predicted class is: ", pred.argmax())  # NOTE: The most important part of the code -> pred.argmax() predict class for input image

            # ----------> All possible prediction values <----------#
            Recognised_character = ""

            if pred.argmax() == 0:
                print("Recognised digit is 0")
                Recognised_character = Recognised_character + "0"

            if pred.argmax() == 1:
                print("Recognised digit is 1")
                Recognised_character = Recognised_character + "1"

            if pred.argmax() == 2:
                print("Recognised digit is 2")
                Recognised_character = Recognised_character + "2"

            if pred.argmax() == 3:
                print("Recognised digit is 3")
                Recognised_character = Recognised_character + "3"

            if pred.argmax() == 4:
                print("Recognised digit is 4")
                Recognised_character = Recognised_character + "4"

            if pred.argmax() == 5:
                print("Recognised digit is 5")
                Recognised_character = Recognised_character + "5"

            if pred.argmax() == 6:
                print("Recognised digit is 6")
                Recognised_character = Recognised_character + "6"

            if pred.argmax() == 7:
                print("Recognised digit is 7")
                Recognised_character = Recognised_character + "7"

            if pred.argmax() == 8:
                print("Recognised digit is 8")
                Recognised_character = Recognised_character + "8"

            if pred.argmax() == 9:
                print("Recognised digit is 9")
                Recognised_character = Recognised_character + "9"

            if pred.argmax() == 10:
                print("Recognised character is A")
                Recognised_character = Recognised_character + "A"

            if pred.argmax() == 11:
                print("Recognised character is B")
                Recognised_character = Recognised_character + "B"

            if pred.argmax() == 12:
                print("Recognised character is C")
                Recognised_character = Recognised_character + "C"

            if pred.argmax() == 13:
                print("Recognised character is D")
                Recognised_character = Recognised_character + "D"

            if pred.argmax() == 14:
                print("Recognised character is E")
                Recognised_character = Recognised_character + "E"

            if pred.argmax() == 15:
                print("Recognised character is F")
                Recognised_character = Recognised_character + "F"

            if pred.argmax() == 16:
                print("Recognised character is G")
                Recognised_character = Recognised_character + "G"

            if pred.argmax() == 17:
                print("Recognised character is H")
                Recognised_character = Recognised_character + "H"

            if pred.argmax() == 18:
                print("Recognised character is I")
                Recognised_character = Recognised_character + "I"

            if pred.argmax() == 19:
                print("Recognised character is J")
                Recognised_character = Recognised_character + "J"

            if pred.argmax() == 20:
                print("Recognised character is K")
                Recognised_character = Recognised_character + "K"

            if pred.argmax() == 21:
                print("Recognised character is L")
                Recognised_character = Recognised_character + "L"

            if pred.argmax() == 22:
                print("Recognised character is M")
                Recognised_character = Recognised_character + "M"

            if pred.argmax() == 23:
                print("Recognised character is N")
                Recognised_character = Recognised_character + "N"

            if pred.argmax() == 24:
                print("Recognised character is O")
                Recognised_character = Recognised_character + "O"

            if pred.argmax() == 25:
                print("Recognised character is P")
                Recognised_character = Recognised_character + "P"

            if pred.argmax() == 26:
                print("Recognised character is Q")
                Recognised_character = Recognised_character + "Q"

            if pred.argmax() == 27:
                print("Recognised character is R")
                Recognised_character = Recognised_character + "R"

            if pred.argmax() == 28:
                print("Recognised character is S")
                Recognised_character = Recognised_character + "S"

            if pred.argmax() == 29:
                print("Recognised character is T")
                Recognised_character = Recognised_character + "T"

            if pred.argmax() == 30:
                print("Recognised character is U")
                Recognised_character = Recognised_character + "U"

            if pred.argmax() == 31:
                print("Recognised character is V")
                Recognised_character = Recognised_character + "V"

            if pred.argmax() == 32:
                print("Recognised character is W")
                Recognised_character = Recognised_character + "W"

            if pred.argmax() == 33:
                print("Recognised character is X")
                Recognised_character = Recognised_character + "X"

            if pred.argmax() == 34:
                print("Recognised character is Y")
                Recognised_character = Recognised_character + "Y"

            if pred.argmax() == 35:
                print("Recognised character is Z")
                Recognised_character = Recognised_character + "Z"

            # ----------> Add the text description to every image <----------#
            addTextWithBackgroundRectangleToImage(img, "Input image")  # Add text and background rectangle on the image
            addTextWithBackgroundRectangleToImage(imgGray, "Grayscale image")
            addTextWithBackgroundRectangleToImage(imgBlur, "Blur image")
            addTextWithBackgroundRectangleToImage(imgThreshold, "Threshold image")
            addTextWithBackgroundRectangleToImage(imgCanny, "Canny image")
            addTextWithBackgroundRectangleToImage(imgDila, "Dilated image")
            addTextWithBackgroundRectangleToImage(imgErode, "Erode image")
            addTextWithBackgroundRectangleToImage(onImgDrawContours, "Contours")
            addTextWithBackgroundRectangleToImage(imgBlank, "Custom text")

            # ----------> Position and display all images at the same time at the same window <----------#
            imgStack = stackImages(round(scalePreviewImagesValue * 0.1, 1), ([img, imgGray, imgBlur],  # Add all the images on the one window
                                                                             [imgThreshold, imgCanny, imgDila], [imgErode, onImgDrawContours, imgBlank]))

            cv2.imshow("Ultimate Computer Vision Script", imgStack)  # Show created OpenCV window

            # ----------> Print the recognised character <----------#
            print("Recognised characters is:", Recognised_character)

            print("---------------------------------------------------------------------")
            print("PRESS 'SPACE' FOR NEXT STEP <=> PRESS 'ESC' two times for end script.")
            print("NOTE: STEP BY STEP mode need to be turned on.")

            # NOTE: If you want to show every detected image step by step uncomment the lines bellow, for the next image you need to press any key, for the exit from the program press "ESC" key two times
            k = cv2.waitKey(0)  # Stops the program until any key is pressed
            if k == 27:  # Listen if the "ESC" key is pressed
                cv2.destroyAllWindows()  # Close all OpenCV windows
                break
        else:  # To this if all contours are smaller of defined contours area
            print("Contour is smaller then defined area")


    if cv2.waitKey(0) == 27:  # Listen if the "ESC" key is pressed
        cv2.destroyAllWindows()  # Close all OpenCV windows
        break  # Beak the while loop = the program is finished
