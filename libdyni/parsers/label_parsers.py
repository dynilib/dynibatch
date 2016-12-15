from os.path import basename, splitext


class CSVLabelParser:
    """csv label files parser
    Args: csv label file(s), written as
        <file_id>,<class>
        <file_id>,<class>
        <file_id>,<class>
        ...
    """

    def __init__(self, *label_files):

        self._labels = {}
        self._label_dict = {}

        for label_file in label_files:
            with open(label_file, "r") as f:
                for line in f:
                    sline = line.split(",")

                    label = sline[1].strip()
                    if not label in self._label_dict:
                        self._label_dict[label] = len(self._label_dict)

                    self._labels[sline[0].strip()] = self._label_dict[label]

    def get_label(self, audio_path):
        return self._labels[splitext(basename(audio_path))[0]]

    def get_labels(self):
        return self._label_dict
