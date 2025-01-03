import cv2
import preprocessing
import documentDetector
import pytesseract
import datetime


# Takes image as input
img = cv2.imread("Input\\input1.jpg")

# Preprocesses image to prepare for img detection
preprocessed_img = preprocessing.preprocesser(img)

# Used to extract document from image
extracted_doc = documentDetector.documentDetector(preprocessed_img)

# If document detection failed, use preprocessed img
if extracted_doc is None:
    print("Document detection failed. Using preprocessed image for OCR.")
    doc_cropped = preprocessed_img

# Convert the cropped/processed image to grayscale (if it's not already)
if len(doc_cropped.shape) == 3:
    doc_cropped = cv2.cvtColor(doc_cropped, cv2.COLOR_BGR2GRAY)

# Use Tesseract to extract text
extracted_text = pytesseract.image_to_string(doc_cropped, lang='eng')

# Generate file name based on current time
current_datetime = datetime.datetime.now()
datetime_string = current_datetime.strftime('%m-%d-%y_%H-%M-%S')
file_name = "output\\" + datetime_string + ".txt"

# Write the read text to the output file
with open(file_name, "w", encoding="utf-8") as f:
    f.write(extracted_text)