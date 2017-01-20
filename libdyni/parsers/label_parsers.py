"""
    Module containing parsers of label files
"""

from os.path import basename, splitext
from libdyni.utils import segment, segment_container


class FileLabelParser:
    pass


class SegmentLabelParser:
    pass


class CSVFileLabelParser(FileLabelParser):
    """csv label files parser
    Args: file2label files, written as
            <file_id><separator><class>
            <file_id><separator><class>
            <file_id><separator><class>
          separator (optional): see above
          label_file: file containing the set of labels to be used
    """

    def __init__(self, *file2label_files, separator=",", label_file=None):

        # get label set
        if label_file:
            with open(label_file, "r") as f:
                self._label_list = set([l.strip() for l in f.readlines() if l.strip()])
        else:
            for file2label_file in file2label_files:
                with open(file2label_file, "r") as f:
                    self._label_list = set([l.split(separator)[1].strip() for l in f.readlines() if l.strip()])

        # sort labels
        self._label_list = sorted(list(self._label_list))

        # create file2label dict
        self._file2label_dict = {}
        for file2label_file in file2label_files:
            with open(file2label_file, "r") as f:
                for line in f:
                    if line:
                        sline = line.split(separator)
                        label = sline[1].strip()
                        self._file2label_dict[sline[0].strip()] = self._label_list.index(label) if label in self._label_list else segment.CommonLabels.unknown

    def get_label(self, audio_path):
        return self._file2label_dict[splitext(basename(audio_path))[0]]

    def get_labels(self):
        return self._label_list


class CSVSegmentLabelParser(SegmentLabelParser):

    def __init__(self,
            seg2label_files_root,
            label_file,
            audio_file_extension=".wav",
            seg_file_extension=".seg",
            seg_file_separator=","):

        self._seg2label_files_root = seg2label_files_root
        self._label_file = label_file
        self._audio_file_extension = audio_file_extension
        self._seg_file_extension = seg_file_extension
        self._seg_file_separator = seg_file_separator
        
        # get label set
        with open(label_file, "r") as f:
            self._label_list = set([l.strip() for l in f.readlines() if l.strip()])

        # sort labels
        self._label_list = sorted(list(self._label_list))

    def get_segment_container(self, audio_path):

        seg_file_path_tuple = (self._seg2label_files_root, audio_path.replace(self._audio_file_extension, self._seg_file_extension))

        return segment_container.create_segment_container_from_seg_file(seg_file_path_tuple,
            self._label_list,
            audio_file_ext=self._audio_file_extension,
            seg_file_ext=self._seg_file_extension,
            seg_file_separator=self._seg_file_separator)
    
    def get_labels(self):
        return self._label_list
