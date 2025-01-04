def parse_words_txt(words_txt_path):
    """
    Returns a dictionary { word_id: text } from words.txt
    Skips lines with segmentation_quality == 'er' (segmentation of word can be bad)
    """

    word_dict = {}
    with open(words_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line.startswith('#') or not line:
                continue  # skip comments and empty lines

            parts = line.split(' ')
            # example line:
            # a01-000u-00-00 ok 154 1 408 768 27 51 AT A
            #  0             1  2   3 4   5   6  7  8  9

            word_id = parts[0]                  # file name, ie. a01-000u-00-00
            segmentation_quality = parts[1]     # segmentation quality, ie. ok or er
            transcription = parts[-1]           # transcription, ie. A

            # Filter out 'er' for only good segmentations
            if segmentation_quality == 'er':
                continue

            word_dict[word_id] = transcription

    return word_dict

def main():
    word_dict = parse_words_txt("archive\\words_new.txt")
    print(word_dict["a01-000u-00-00"])  # should print "A"

if __name__ == "__main__":
    main()