import cv2
import numpy as np
from keras.models import load_model # Used for load trained CNN model

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

def getContours(img,onImgDrawContours,fromImgExtractObject,iterationCounter):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # Finding all contours on the image
    for cnt in contours:                 # Individually access to every contour
        area = cv2.contourArea(cnt)      # Calculate area of every image
        print("Contours area: ",area)

        if area>300:
            cv2.drawContours(onImgDrawContours, cnt, -1, (255, 0, 0), 3) # Draw contours "cnt" on the image "imageContour"
            peri = cv2.arcLength(cnt,True)                        # Calculate the acr length of the contours
            #print(peri)                                          # Print the acr length of the contours
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)         # Looking for number of edges on the contour
            print("Number of edges for selected contour: ",len(approx))                                    # Print the number of edges on the contour
            objCor = len(approx)                                  # Store the number of edges on the contour in a variable
            x, y, w, h = cv2.boundingRect(approx)                 # Returns coordinates, width and height for drawing bounding box around contour

            if objCor ==3: objectType ="Tri"                               # Classificator related with the number od contour's edges
            elif objCor == 4:                                              # Classificator related with the number od contour's edges
                aspRatio = w/float(h)                                      # Square and rectangle classificator
                if aspRatio >0.98 and aspRatio <1.03: objectType= "Square"
                else:objectType="Rectangle"
            elif objCor>4: objectType= "Circles"
            else:objectType="None"

            text_width_rectangle_offest = 1.1
            text_height_rectangle_offest = 1.7
            textScale = 0.5
            textThickness=1
            putTextOffest = textThickness + 3
            boundingBoxRectangleThickness=2

            wantedText = "Selection"
            startCoordinateX =x
            startCoordinateY =y

            cv2.rectangle(onImgDrawContours, (startCoordinateX, startCoordinateY), (startCoordinateX + w, startCoordinateY +h), (0, 255, 0), boundingBoxRectangleThickness)  # Draw the rectangle around the contour
            (test_width, text_height), baseline = cv2.getTextSize(wantedText, cv2.FONT_HERSHEY_SIMPLEX, textScale,textThickness)  # cv.getTextSize(	text, fontFace, fontScale, textThickness) -> output=((189, 22), 10)
            #print('BB text size: ', test_width, text_height)
            cv2.rectangle(onImgDrawContours, (startCoordinateX-boundingBoxRectangleThickness, startCoordinateY), (startCoordinateX + int(text_width_rectangle_offest * test_width),startCoordinateY + int(text_height_rectangle_offest * text_height)), (0, 255, ), cv2.FILLED)
            cv2.putText(onImgDrawContours, wantedText,(startCoordinateX + putTextOffest, startCoordinateY + text_height + putTextOffest),cv2.FONT_HERSHEY_SIMPLEX, textScale, (0, 0, 0), textThickness)



            if w>h:
                extend_inital_object_selection_percent=int(0.1 * w)
            else:
                extend_inital_object_selection_percent=int(0.1 * h)


            text_width_rectangle_offest = 1.1
            text_height_rectangle_offest = 1.7
            textScale=0.5
            textThickness=1
            putTextOffest = textThickness + 3
            boundingBoxRectangleThickness=2

            wantedText = "Extended selection"
            startCoordinateX =x - int(extend_inital_object_selection_percent/2)
            startCoordinateY =y +h + int(extend_inital_object_selection_percent/2)

            cv2.rectangle(onImgDrawContours, (x - int(extend_inital_object_selection_percent/2), y - int(extend_inital_object_selection_percent/2)), (x + w + int(extend_inital_object_selection_percent/2), y + h + int(extend_inital_object_selection_percent/2)), (0, 255, 0), boundingBoxRectangleThickness)  # Draw the rectangle around the contour
            (test_width, text_height), baseline = cv2.getTextSize(wantedText, cv2.FONT_HERSHEY_SIMPLEX, textScale,textThickness)  # cv.getTextSize(	text, fontFace, fontScale, textThickness) -> output=((189, 22), 10)
            #print('BB text size: ', test_width, text_height)
            cv2.rectangle(onImgDrawContours, (startCoordinateX-boundingBoxRectangleThickness, startCoordinateY), (startCoordinateX + int(text_width_rectangle_offest * test_width),startCoordinateY + int(text_height_rectangle_offest * text_height)), (0, 255, ), cv2.FILLED)
            cv2.putText(onImgDrawContours, wantedText,(startCoordinateX + putTextOffest, startCoordinateY + text_height + putTextOffest),cv2.FONT_HERSHEY_SIMPLEX,textScale, (0, 0, 0), textThickness)

            ############ Crop selected object
            boundingBoxStart_EndCoordinates=[x - int(extend_inital_object_selection_percent/2),y - int(extend_inital_object_selection_percent/2),x + w + int(extend_inital_object_selection_percent/2),y + h + int(extend_inital_object_selection_percent/2)]

            imgObjectSelected=fromImgExtractObject[boundingBoxStart_EndCoordinates[1]:boundingBoxStart_EndCoordinates[3],boundingBoxStart_EndCoordinates[0]:boundingBoxStart_EndCoordinates[2]] # First height range, then width range
            #print("imgObjectSelected:",imgObjectSelected)
            print("Number of iteration ---------------> ",iterationCounter)

            cv2.imshow("Image of object", imgObjectSelected)
            print("Selected object shape: ",imgObjectSelected.shape )

            ########## Scale selected object
            heght=imgObjectSelected.shape[0]
            width=imgObjectSelected.shape[1]

            cv2.imshow("imgObjectSelected", imgObjectSelected)  # Display image
            print("imgObjectSelected shape:", imgObjectSelected.shape)  # Print the image shape in console
            heght = imgObjectSelected.shape[0]  # Get input image height
            width = imgObjectSelected.shape[1]  # Get input image width

            # ---------> Resize input image to wanted dimension <--------#
            if width > heght:  # Looking for the maximal dimension of input image
                maxSelectedObjectDimension = width
            else:
                maxSelectedObjectDimension = heght

            wantedMaxSelectedObjImageDimension = 28  # Define maximal dimension after scaling
            scalingRate = maxSelectedObjectDimension / wantedMaxSelectedObjImageDimension  # Get scale ratio
            # print("Scalerate is:", scalingRate)                                                # Print the sale rate
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
            # NOTE: np.invert is giving batchsize dimension to image. e.g (28,28)->(1,28,28)
            imgObjectSelectedResizedClrInverted = np.invert(np.array([imgObjectSelectedResized]))  # Inverting image pixels value e.g: black color to white or reverse
            print('imgObjectSelectedResizedClrInverted', imgObjectSelectedResizedClrInverted.shape)  # Print image shape
            # print('imgObjectSelectedResizedClrInverted', imgObjectSelectedResizedClrInverted)       # Shows every image pixel value

            if heght > width:
                makeImgToImgOffsetOnAxsisX = int((heght - width) / 2)
                makeImgToImgOffsetOnAxsisY = 0
            else:
                makeImgToImgOffsetOnAxsisY = int((width - heght) / 2)
                makeImgToImgOffsetOnAxsisX = 0

            # ---------> Adding black background in order to get 28x28 image which will be ready for input in CNN <--------#
            blackBcgImg = cv2.imread("../developmental_phase_images/Black_background.png")[:, :, 0]  # Load image with one channel "grayscale"
            cv2.imshow("blackBcgImg", blackBcgImg)  # Display image
            print("blackBcgImg:", blackBcgImg.shape)
            # NOTE: Upper left corner of every image has coordinates 0,0
            drawImgToImagStartCordinateX = makeImgToImgOffsetOnAxsisX  # X coordinate where will start our inserting image
            drawImgToImagStartCordinateY = makeImgToImgOffsetOnAxsisY  # Y coordinate where will start our inserting image
            blackBcgImg[drawImgToImagStartCordinateY: drawImgToImagStartCordinateY + heght,drawImgToImagStartCordinateX: drawImgToImagStartCordinateX + width] = imgObjectSelectedResizedClrInverted  # Paste image in to another image
            print("imgToImgCoordinates: ", drawImgToImagStartCordinateY, drawImgToImagStartCordinateY + heght,drawImgToImagStartCordinateX, drawImgToImagStartCordinateX + width)
            cv2.imshow('imgToImg', blackBcgImg)
            print("imgToImg", blackBcgImg.shape)
            # NOTE: Shape needed for CNN input is (1,28,28)
            imgGoToCnn = np.reshape(blackBcgImg, (1,28,28,1))
            print("imgGoToCnn", imgGoToCnn.shape)
            # NOTE: Needed values normalisation before the image goes in CNN
            imgGoToCnn = imgGoToCnn / 255.0

            pred = model.predict(imgGoToCnn)
            print("Predicted value is : ")
            print(pred.argmax())
 ##########################################################################
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

 ##########################################################################
            print("Recognised characters is: ",Recognised_character)


            k=cv2.waitKey(0)
            if k == 27:
                cv2.destroyAllWindows()
                break



cv2.namedWindow('Trackbars')
cv2.createTrackbar('LVal', 'Trackbars', 0, 255, nothing)  # Create trackbar 1
cv2.createTrackbar('UVal', 'Trackbars', 0, 255, nothing)  # Create trackbar 1
#cv2.setTrackbarPos( 'LVal', 'Trackbars', 30)  # Set default trackbars values
#cv2.setTrackbarPos( 'UVal', 'Trackbars', 90)  # Set default trackbars values

iterationCounter=0

# load model
model = load_model(
    '../../4.0_developmental_phase_CNN_models/2.0_saved_trained_models/Model_Batchsize=500__Epoch=60__Accuracy=77.5.h5')
# summarize model.
model.summary()

while True:

    # NOTE: ".." in the path means go to the parent's directory
    path= '../../developmental_phase_images/H_S_U.png'  # Wanted image
    img = cv2.imread(path)                                  # Read the wanted image
    #print("Input image shape:", img.shape)
    imgContour = img.copy()                               # Copy the wanted image, on this one we will draw the bounding-boxes

    LVal = cv2.getTrackbarPos('LVal', 'Trackbars')  # Read value from trackbar
    UVal = cv2.getTrackbarPos('UVal', 'Trackbars')  # Read value from trackbar

    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #Contrast
    imgBlur =cv2.GaussianBlur(imgGray,(7,7),1)
    imgTreshold=cv2.threshold(imgBlur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    imgCanny = cv2.Canny(imgTreshold,LVal,UVal)
    iterationCounter =iterationCounter+1
    getContours(imgCanny, imgContour, imgTreshold,iterationCounter)  # Call the function for the contours processing
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel,iterations=2)
    imgErode = cv2.erode(imgDial,kernel,iterations=1)
    imgBlank = np.zeros_like(img)                   # Fill the stack image free place



    addTextWithBackroundRectangleToImage(img,"Input image")
    addTextWithBackroundRectangleToImage(imgGray,"Grayscale image")
    addTextWithBackroundRectangleToImage(imgBlur,"Blur image")
    addTextWithBackroundRectangleToImage(imgTreshold,"Treshold image")
    addTextWithBackroundRectangleToImage(imgCanny,"Canny image")
    addTextWithBackroundRectangleToImage(imgDial,"Dilatated image")
    addTextWithBackroundRectangleToImage(imgErode,"Erode image")
    addTextWithBackroundRectangleToImage(imgContour,"Contours")
    addTextWithBackroundRectangleToImage(imgBlank, "Noting")
    addTextWithBackroundRectangleToImage(imgBlank, "Noting")




    imgStack = stackImages(0.4,([img,imgGray,imgBlur],[imgTreshold,imgCanny,imgDial],[imgErode,imgContour,imgBlank]))
    cv2.imshow("Ultimate Computer Vision Script", imgStack)


    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
        break

cv2.destroyAllWindows()

