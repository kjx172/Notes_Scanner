import cv2
import numpy as np

def documentDetector(img):
    """
    Attempts to locate the largest quadrilateral in the image (which would be the notebook in this case) 
    Performs a prospective transformation and returns the warped/cropped document, returns none if no document found
    """

    # Blur image (might remove due to previous thresholding in preprocessing)
    kernel_size = 5
    blur_img = cv2.GaussianBlur(img,(kernel_size, kernel_size),0)

    # Find the edges in the image
    edges = cv2.Canny(blur_img,150,250,apertureSize = 3)

    # CLose any gaps in the edges
    kernel = np.ones((5,5), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    #cv2.imwrite('output/closed_edges.jpg', closed_edges)

    # Find the contours in closed edges
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Use largest contour to approximate the contour to a polygon
    largest_contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest_contour, True)
    approx_poly = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)

    # If there arent 4 corners the notebook shape wasnt detected
    if len(approx_poly) != 4:
        #print("No 4-corner contour detected; skipping perspective transform.")
        return None
    
    # Reorder corners to follow top-left, top-right, bottom-right, and bottom-left order
    doc_corners = reorder_corners(approx_poly.reshape(4, 2))

    # Perform perspective transform
    warped = four_point_transform(img, doc_corners)
    cv2.imwrite('output/document_cropped.jpg', warped)

    return warped

def reorder_corners(corners):
    """
    Reorder corners to [top-left, top-right, bottom-right, bottom-left].
    """

    # Separate into x and y lists
    rect = np.zeros((4, 2), dtype="float32")

    s = corners.sum(axis=1)
    diff = np.diff(corners, axis=1)

    # top-left: smallest sum
    rect[0] = corners[np.argmin(s)]
    # bottom-right: largest sum
    rect[2] = corners[np.argmax(s)]
    # top-right: smallest difference
    rect[1] = corners[np.argmin(diff)]
    # bottom-left: largest difference
    rect[3] = corners[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    """
    Given an image and four corner points [tl, tr, br, bl],
    returns a warped (top-down) perspective transform.
    """
    (tl, tr, br, bl) = pts

    # Compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Destination points for the transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped
