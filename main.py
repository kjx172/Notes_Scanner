import cv2
import preprocessing

# Takes image as input
img = cv2.imread("Input\skewed_input.png")

# Calls the pre processor
preprocessing.preprocesser(img)