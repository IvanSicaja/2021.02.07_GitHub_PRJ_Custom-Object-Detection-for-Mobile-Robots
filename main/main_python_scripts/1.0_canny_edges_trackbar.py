import cv2


def nothing(x):
	pass


cv2.namedWindow('window')
cv2.createTrackbar('lower', 'window', 0, 255, nothing)
cv2.createTrackbar('upper', 'window', 0, 255, nothing)
cv2.setTrackbarPos('lower', 'window', 30)  # Set default trackbars values
cv2.setTrackbarPos('upper', 'window', 90)  # Set default trackbars values


path= '../../developmental_phase_images/H.png'  # Wanted image
img = cv2.imread(path)                              # Read the wanted image


#cap = cv2.VideoCapture(0)


while True:
	#_, img = cap.read()

	# img = cv2.blur(img, (3,3))

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	x = cv2.getTrackbarPos('lower', 'window')
	y = cv2.getTrackbarPos('upper', 'window')

	imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
	# NOTE: If you are using the Treshold layer before the Canny layer, you won't see slider interaction
	# imgTreshold = cv2.threshold(imgBlur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
	imgCanny = cv2.Canny(imgBlur, x, y)


	cv2.imshow('window', imgCanny)

	if cv2.waitKey(1) == 27:
		cv2.destroyAllWindows()
		break