import cv2
import preprocessing
import documentDetector
import pytesseract
import os

def main():
    exit_program = False

    while not exit_program:
        # Get the class from within the input folder
        print("========================================")
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

            # Convert the cropped/processed image to grayscale (if it's not already)
            if len(doc_cropped.shape) == 3:
                doc_cropped = cv2.cvtColor(doc_cropped, cv2.COLOR_BGR2GRAY)

            # Use Tesseract to extract text
            extracted_text = pytesseract.image_to_string(doc_cropped, lang='eng')

            # Generate file name based on input file name
            clipped_input_name = os.path.splitext(os.path.basename(file_path))[0]
            output_file_name = output_path + "\\" + clipped_input_name +  "_output" + ".txt"

            # Write the read text to the output file
            with open(output_file_name, "w", encoding="utf-8") as f:
                f.write(extracted_text)

        print("Files have been scanned")
        print("========================================")

if __name__ == "__main__":
    main()