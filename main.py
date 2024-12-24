import cv2
import preprocessing

# Takes image as input
img = cv2.imread("Input\\input2.jpg")

# For just saving image to file without placing in note preprocessing is unecissary

# Calls the pre processor
preprocessing.preprocesser(img)

