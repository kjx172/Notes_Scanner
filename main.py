import cv2
import preprocessing
import documentDetector
from kraken import pageseg, rpred
from kraken.lib import models
from PIL import Image
import os

# Load Kraken model (loaded here to avoid repeatedly reloading).
rec_model_path = '/path/to/recognition/model'
model = models.load_any(rec_model_path)


def main():
    exit_program = False

    print("========================================")
    while not exit_program:
        # Get the class from within the input folder
        print("(Enter EXIT to end program.)")
        input_folder_name = input("Enter the name of the folder in the input folder you would like to scan: ")
        
        if input_folder_name == "EXIT":
            print("Goodbye")
            break

        folder_path = "Input\\" + input_folder_name

        # If folder not found, keep trying for correct input
        while not os.path.exists(folder_path):
            print("========================================")
            print("Folder not found, please try again.")
            print("(Enter EXIT to end program.)")
            input_folder_name = input("Enter the name of the folder in the input folder you would like to scan: ")

            if input_folder_name == "EXIT":
                print("Goodbye")
                exit_program = True
                break

            folder_path = "Input\\" + input_folder_name

        # If exit prompt entered while trying for input exit the program
        if exit_program:
            break

        # Create output path for the input folder if it doesnt exist
        output_path = "Scanned_Notes\\" + input_folder_name
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        # For each file in the inputed class folder scan the text
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            # Takes image as input
            img = cv2.imread(file_path)

            # Preprocesses image to prepare for img detection
            preprocessed_img = preprocessing.preprocesser(img)

            # Used to extract document from image
            extracted_doc = documentDetector.documentDetector(preprocessed_img)

            # If document detection failed, use preprocessed img
            if extracted_doc is None:
                #print("Document detection failed. Using preprocessed image for OCR.")
                doc_cropped = preprocessed_img

            # Convert to PIL image for kraken use
            pil_img = Image.fromarray(doc_cropped)

            # Run Kraken OCR
            extracted_text = kraken_ocr(pil_img)

            # Generate file name based on input file name
            clipped_input_name = os.path.splitext(os.path.basename(file_path))[0]
            output_file_name = output_path + "\\" + clipped_input_name +  "_output" + ".txt"

            # Write the read text to the output file
            with open(output_file_name, "w", encoding="utf-8") as f:
                f.write(extracted_text)

        print("Files have been scanned")
        print("========================================")

def kraken_ocr(pil_image):
    """
    Performs line segmentation and OCR using Kraken on a PIL Image.
    Returns the recognized text as a single string.
    """

    # Segment page into lines
    segmentation = pageseg.segment(pil_image)

    # Recognize text line by line
    lines = rpred.rpred(model, pil_image, segmentation)

    # 4. Combine recognized lines into a single string
    recognized_text = "\n".join(line["text"] for line in lines)
    return recognized_text


if __name__ == "__main__":
    main()