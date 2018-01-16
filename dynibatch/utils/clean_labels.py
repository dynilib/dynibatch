import csv
import argparse
from dynibatch.utils.segment_container import create_segment_containers_from_audio_files
from dynibatch.parsers.label_parsers import CSVFileLabelParser


def generate_cleanner_labels_csv(folder_data, file2label_in, file2label_out, label_file):
    """ Create a csv file based on file2label_in containing only the file in folder_data.
    (see CSVFileLabelParser)

    Args:
        folder_data (str): folder path containing the data
        file2label_in (str): input file2label file
        file2label_out (str): output file2label file
        label_file (str): label file
    """
    sc_gen = create_segment_containers_from_audio_files(folder_data)
    parser = CSVFileLabelParser(file2label_in, label_file=label_file)

    with open(file2label_out, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for sc in sc_gen:
            spamwriter.writerow([sc.audio_path, parser.get_label(sc.audio_path)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean csv with only existing files')
    parser.add_argument('folder_data', metavar='folder_data', type=str,
                        help='folder where are the audio files')
    parser.add_argument('file2label_in', metavar='file2label_in', type=str,
                        help='file2label file to clean')
    parser.add_argument('file2label_out', metavar='file2label_out', type=str,
                        help='cleaned file2label file')
    parser.add_argument('label_file', metavar='label_file', type=str,
                        help='label file')
    args = parser.parse_args()

    generate_cleanner_labels_csv(args.folder_data,
                                 args.file2label_in,
                                 args.file2label_out,
                                 label_file)
