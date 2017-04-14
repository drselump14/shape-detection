# USAGE
# python detect_shapes.py --image shapes_and_colors.png

# import the necessary packages
from pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2
import pdb
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
image = cv2.imread(args["image"])
resized = imutils.resize(image, width=300)
# pdb.set_trace()
ratio = image.shape[0] / float(resized.shape[0])

lower_white = np.array([220, 220, 220], dtype=np.uint8)
upper_white = np.array([255, 255, 255], dtype=np.uint8)
mask = cv2.inRange(resized, lower_white, upper_white)  # cou
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
mask = cv2.bitwise_not(mask)
# cv2.imshow("Image", mask)
# cv2.waitKey(0)

# load background (could be an image too)
bk = np.full(resized.shape, 255, dtype=np.uint8)  # white bk
# get masked foreground
fg_masked = cv2.bitwise_and(resized, resized, mask=mask)
cv2.imshow("Image", fg_masked)
cv2.waitKey(0)

# get masked background, mask must be inverted
# mask = cv2.bitwise_not(mask)
# bk_masked = cv2.bitwise_and(bk, bk, mask=mask)
# cv2.imshow("Image", bk_masked)
# cv2.waitKey(0)

# convert the resized image to grayscale, blur it slightly,
# and threshold it
gray = cv2.cvtColor(fg_masked, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
cv2.imshow("Image", thresh)
cv2.waitKey(0)
# find contours in the thresholded image and initialize the
# shape detector
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
sd = ShapeDetector()

# loop over the contours
for c in cnts:
	# compute the center of the contour, then detect the name of the
	# shape using only the contour
	M = cv2.moments(c)
	cX = int((M["m10"] / M["m00"]) * ratio)
	cY = int((M["m01"] / M["m00"]) * ratio)
	shape = sd.detect(c)

	# multiply the contour (x, y)-coordinates by the resize ratio,
	# then draw the contours and the name of the shape on the image
	c = c.astype("float")
	c *= ratio
	c = c.astype("int")
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
	cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
		0.5, (0,0,0), 2)

	# show the output image
	cv2.imshow("Image", image)
	cv2.waitKey(0)
