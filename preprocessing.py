import cv2
import numpy as np
from scipy.ndimage import interpolation as inter

def correctSkew(image, delta = 1, limit = 10):
    """
    Corrects skew in the input image for about +/- 10 degrees (Increase limit for larger degree).
    
    Parameters:
        image: Input image in which skew needs to be corrected.
        delta: Step size for angle increments during the search for the best angle.
        limit: Range of angles to search in both directions (positive and negative).
    
    Returns:
        corrected: The skew-corrected image.
    """

    def determine_score(arr, angle):
        """
        Computes a score for how aligned the text in the image is at a given angle.

        Parameters:
            arr: Binary image array (e.g., thresholded image).
            angle: Angle to rotate the image for scoring.
        
        Returns:
            histogram: Sum of pixels along each row after rotation.
            score: Metric indicating the alignment of the text.
        """

        # Rotate image by given angle without changing dimensions
        data = inter.rotate(arr, angle, reshape = True, order = 0)

        # Compute the sum of pixel values along each row (horrizontal projection)
        histogram = np.sum(data, axis = 1, dtype = float)

        # Calculate allignment score based on differences between adjacent rows
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype = float)
        return score

    # Converts image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    # Apply thresholding using Otsu's method
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Initialize list of scores for each angle
    scores = []

    # Generates a range of test angles to test for skew correction and computes the score for each angle
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        score = determine_score(thresh, angle)
        scores.append(score)

    # The angle with the maximum score has the best allignment
    best_angle = angles[scores.index(min(scores))]

    # Calculate center of image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Create rotation matrix M for the best angle and use it to rotate image
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
            borderMode=cv2.BORDER_REPLICATE)
    
    return corrected_image

def preprocesser(input_image):
    # Normalize image to between 0 and 255 (Creates uniform range for different input images)
    norm_img = np.zeros((input_image.shape[0], input_image.shape[1]))
    normalized_img = cv2.normalize(input_image, norm_img, 0, 255, cv2.NORM_MINMAX)

    # Skew correction
    unskewed_img = correctSkew(normalized_img)

    cv2.imshow('unskewed_img',unskewed_img)
    cv2.waitKey(0)
    