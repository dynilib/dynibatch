import csv
import argparse
from dynibatch.utils.segment_container import create_segment_containers_from_audio_files
from dynibatch.parsers.label_parsers import CSVFileLabelParser


def generate_cleanner_labels_csv(folder_data, label_file_in, label_file_out):
    """ Create a csv file based on label_file_in containing only the file in folder_data.

    Args:
        folder_data (str): folder path containing the data
        label_file_in (str): file path containing all the labels
        label_file_out (str): file path containing the new labels
    """
    sc_gen = create_segment_containers_from_audio_files(folder_data)
    parser = CSVFileLabelParser(label_file_in)

    with open(label_file_out, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for sc in sc_gen:
            spamwriter.writerow([sc.audio_path, parser.get_label(sc.audio_path)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean csv with only existing files')
    parser.add_argument('folder_data', metavar='folder_data', type=str,
                        help='folder where is the audios')
    parser.add_argument('label_file_in', metavar='label_file_in', type=str,
                        help='label file to clean')
    parser.add_argument('label_file_out', metavar='label_file_out', type=str,
                        help='label file path to create')
    args = parser.parse_args()

    generate_cleanner_labels_csv(args.folder_data,
                                 args.label_file_in,
                                 args.label_file_out)
