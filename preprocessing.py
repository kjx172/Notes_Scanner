import cv2
import numpy as np
from wand.image import Image as WandImage
from wand.color import Color
from PIL import Image as PILImage


def correctSkew(input_image):

    # Converts input from np array into wand image object
    wand_img = WandImage.from_array(input_image.astype(np.uint8), channel_map="BGR")

    # Removes the skew from the input image
    wand_img.deskew(0.4*wand_img.quantum_range)

    # Converts wand image object back into np array
    with wand_img:
        wand_img.background_color = Color('white')
        wand_img.format = 'jpg'
        wand_img.alpha_channel = False

        # Fill image buffer with numpy array from blob
        img_buffer=np.asarray(bytearray(wand_img.make_blob()), dtype=np.uint8)

    # Return the cv image (np array)
    if img_buffer is not None:
        return cv2.imdecode(img_buffer, cv2.IMREAD_UNCHANGED)
    
    return input_image

def scaleImage(input_image):
    orig_width, orig_height = input_image.shape[1], input_image.shape[0]

    # If original size is smaller than reccomended size
    if (orig_width < 2550 and orig_height < 3300):
        input_image = cv2.resize(input_image, (2550, 3300), interpolation = cv2.INTER_CUBIC)
    
    return input_image

def preprocesser(input_image):
    # Normalize image to between 0 and 255 (Creates uniform range for different input images)
    norm_img = np.zeros((input_image.shape[0], input_image.shape[1]), dtype=np.uint8)
    normalized_img = cv2.normalize(input_image, norm_img, 0, 255, cv2.NORM_MINMAX)

    #cv2.imwrite("output/normalized_img.jpg", normalized_img)

    # Skew correction
    unskewed_img = correctSkew(normalized_img)

    #cv2.imwrite("output/unskewed_img.jpg", unskewed_img)

    # Scale the image if under the reccomended pixel size for OCR for A4 letter paper
    # May have to change depending if the scaling ruins the legibility
    scaled_img = scaleImage(unskewed_img)
    #cv2.imwrite("output/scaled_img.jpg", scaled_img)

    # Convert to grayscale before denoising
    if len(scaled_img.shape) == 3:
        gray_scaled = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_scaled = scaled_img.copy()

    # Removes any noise from the image
    denoised_gray = cv2.fastNlMeansDenoising(gray_scaled, None, 10, 7, 21)
    #cv2.imwrite("output/denoised_img.jpg", denoised_gray)

    # Thinning and Skeletization
    # Eroding image may ruin legibility
    kernel = np.ones((3,3),np.uint8)
    eroded_img = cv2.erode(denoised_gray, kernel, iterations = 1)

    #cv2.imwrite("output/eroded_img.jpg", eroded_img)

    # Thresholds image
    _, thresh_img = cv2.threshold(eroded_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #cv2.imwrite("output/thresh_img.jpg", thresh_img)


    return thresh_img
    