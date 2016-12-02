from os.path import join, basename, splitext
import xmltodict


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
        for label_file in label_files:
            with open(label_file, "r") as f:
                for line in f:
                    sline = line.split(",")
                    self._labels[sline[0].strip()] = sline[1].strip()

    def get_label(self, audio_path):
        return self._labels[splitext(basename(audio_path))[0]]


class Bird2016Parser:

    def __init__(self, label_root):
        self._label_root = label_root

    def get_label(self, audio_path):
        label_path = join(self._label_root,
                          splitext(basename(audio_path))[0] + ".xml")
        with open(label_path, "rb") as f:
            return xmltodict.parse(f)["Audio"]["ClassId"]

# TODO: sermicro
#def get_label(audio_path):
#    return os.basename(audio_path).split("_")[0]
