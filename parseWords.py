import os

def parse_words_txt(words_txt_path):
    """
    Returns a dictionary { word_id: text } from words_new.txt
    Skips lines with segmentation_quality == 'err' (segmentation of word can be bad)
    """

    word_dict = {}
    with open(words_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line.startswith('#') or not line:
                continue  # skip comments and empty lines

            parts = line.split(' ')
            # example line: a01-000u-00-00 ok 154 1 408 768 27 51 AT A

            word_id = parts[0]                  # file name, ie. a01-000u-00-00
            segmentation_quality = parts[1]     # segmentation quality, ie. ok
            transcription = parts[-1]           # transcription, ie. A

            # Filter out 'err' for only good segmentations
            if segmentation_quality == 'err':
                continue

            word_dict[word_id] = transcription

    return word_dict

def create_word_image_pairs(word_dict, root):
    """
    Returns a list of tuples: [(image_path, transcription), ...]
    for each valid image that appears in word_dict.
    """
    pairs = []
    
    # Loops through the folders containing the images
    for subdir, dirs, files, in os.walk(root):
        for filename in files:
            if filename.endswith('.png'):
                # Example: a01-000u-00-00.png
                base_name = os.path.splitext(filename)[0]  # "a01-000u-00-00"

                if base_name in word_dict:
                    full_path = os.path.join(subdir, filename)
                    text = word_dict[base_name]
                    pairs.append((full_path, text))

    return pairs

def save_pairs(pairs):

    # Make a directory to store the pairs
    output_dir = "kraken_word_pairs"
    os.makedirs(output_dir, exist_ok=True)

    for i, (img_path, text) in enumerate(pairs):
        base = f"sample_{i}"
        # Copy or symlink the image
        new_img_path = os.path.join(output_dir, base + ".png")

        # Write the text in a .txt file
        txt_path = os.path.join(output_dir, base + ".gt.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

def main():
    # Get the file name + word from the words_new file
    word_dict = parse_words_txt("archive\\words_new.txt")

    # Create a tuple containing the image path and ground truth
    pairs = create_word_image_pairs(word_dict, "archive\\words")

    # Save pairs to a file to use to train model
    save_pairs(pairs)


if __name__ == "__main__":
    main()