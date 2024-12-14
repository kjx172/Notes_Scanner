import cv2
import numpy as np
from wand.image import Image as WandImage
from wand.color import Color


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


def preprocesser(input_image):
    # Normalize image to between 0 and 255 (Creates uniform range for different input images)
    norm_img = np.zeros((input_image.shape[0], input_image.shape[1]))
    normalized_img = cv2.normalize(input_image, norm_img, 0, 255, cv2.NORM_MINMAX)

    # Skew correction
    unskewed_img = correctSkew(normalized_img)

    cv2.imshow('unskewed_img',unskewed_img)
    cv2.waitKey(0)
    