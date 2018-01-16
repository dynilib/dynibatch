#The MIT License
#
#Copyright (c) 2017 DYNI machine learning & bioacoustics team - Univ. Toulon
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of
#this software and associated documentation files (the "Software"), to deal in
#the Software without restriction, including without limitation the rights to
#use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
#the Software, and to permit persons to whom the Software is furnished to do so,
#subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
#FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
#IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
#CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


from dynibatch.utils import segment, segment_container
from dynibatch.utils.exceptions import ParsingError


class FileLabelParser:
    pass


class SegmentLabelParser:
    pass


def parse_label_file(path, separator):
    # Parse label file written as:
    #
    #       <label_id><separator><label_name>
    #       <label_id><separator><label_name>
    #       <label_id><separator><label_name>
    #       ...
    #
    # Returns a dictionary with <label_id>:<label_name> mapping

    label_dict = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            _id, name = line.split(separator)
            _id = int(_id)
            name = name.strip()
            if _id < 0:
                raise ParsingError("{}: Label id must be greater or equal to 0".format(label_file))
            if _id in label_dict.keys():
                raise ParsingError("{}: Label id {} already used".format(label_file, _id))
            if name in label_dict.values():
                raise ParsingError("{}: Label name {} already used".format(label_file, name))
            label_dict[_id] = name
    return label_dict


def parse_file2label_file(paths, separator, label_dict):
    # Parse file2label file written as:
    #
    #       <filename><separator><label_id>
    #       <filename><separator><label_id>
    #       <filename><separator><label_id>
    #       ...
    #    
    # Returns a dictionary with <filename>:<label_id> mapping

    file2label_dict = {}
    for path in paths:
        with open(path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                filename, label_id = line.split(separator)
                filename = filename.strip()
                label_id = int(label_id)
                if filename in file2label_dict.keys() and file2label_dict[filename] != label_id:
                    raise ParsingError("{}: File {} found multiple times with different labels".format(
                        path, filename))
                file2label_dict[filename] = label_id if label_id in label_dict.keys() \
                    else segment.CommonLabels.unknown.value
    return file2label_dict


class CSVFileLabelParser(FileLabelParser):
    """File-based label file parser (1 audio file = 1 label).

    This object parses CSV files (aka "file2label" files) written in the following format:

            <file_path><separator><label_id>
            <file_path><separator><label_id>
            <file_path><separator><label_id>
            ...

    where <file_path> is the path, relative to some root path, of an audio file
    (see test/data/file2label.csv for an example).

    The label_file argument constrain the set of labels to
    be used. This file is a simple list of label ids and names, i.e.:

            <id>,<name>
            <id>,<name>
            <id>,<name>
            ...

    All files with a label id which is not in the list specified in
    label_file will have their label id set to segment.CommonLabels.unknown.value.
    """

    def __init__(self, *file2label_files, label_file, separator=","):
        """Create a label dict and a file2label dict to map a file to a label.

        Args:
            file2label_files (*str): one or several file2label files.
            label_file (str): file containing the list of label ids and names.
            separator (str): character used as a separator in the
                file2label file
         """

        self._label_dict = parse_label_file(label_file, separator)
        self._file2label_dict = parse_file2label_file(file2label_files, separator, self._label_dict)

    def get_label(self, audio_path):
        """Returns the label of audio_path

        Args:
            audio_path: (relative) audio path
        """
        return self._file2label_dict[audio_path]

    def get_labels(self):
        """Returns the dict of labels"""
        return self._label_dict


class CSVSegmentLabelParser(SegmentLabelParser):
    """Segment-based label file parser (1 segment = 1 label).

    This object parses CSV files (aka "seg2label" files) written in the following format:

            <start_time><separator><end_time><separator><label_id>
            <start_time><separator><end_time><separator><label_id>
            <start_time><separator><end_time><separator><label_id>
            ...

    where <start_time> and <end_time> are given in seconds (see test/data/*.seg
    for some examples).

    Every audio file must have a corresponding seg2label file in a
    seg2label_files_root. seg2label_files_root must have the same structure as
    the audio files root.

    A label_file argument must be set to specify the set of labels to
    be used. This file is a simple list of label ids and names, i.e.:

            <id>,<name>
            <id>,<name>
            <id>,<name>
            ...

    All segments with a label which is not in the list specified in label_file
    will have their label set to segment.CommonLabels.unknown.value.
    """

    def __init__(self,
                 seg2label_files_root,
                 label_file,
                 audio_file_extension=".wav",
                 seg_file_extension=".seg",
                 separator=","):
        """Create a label dict.

        Args:
            seg2label_files_root (str): root path of the seg2label files.
            label_file: file containing the list of labels to be
                used.
            audio_file_extension (str)
            seg_file_extension (str)
            separator (str): csv file separator (for seg2label files and label_file)
         """

        self._seg2label_files_root = seg2label_files_root
        self._label_file = label_file
        self._audio_file_extension = audio_file_extension
        self._seg_file_extension = seg_file_extension
        self._separator = separator

        # get label set
        self._label_dict = parse_label_file(label_file, separator)

    def get_segment_container(self, audio_path):
        """Returns a segment container with all the segments set to the labels
        specified in the seg2label files

        Args:
            audio_path

        Returns:
            SegmentContainer
        """

        seg_file_path_tuple = (self._seg2label_files_root,
                               audio_path.replace(self._audio_file_extension,
                                                  self._seg_file_extension))

        return segment_container.create_segment_container_from_seg_file(
            seg_file_path_tuple,
            self._label_dict,
            audio_file_ext=self._audio_file_extension,
            seg_file_ext=self._seg_file_extension,
            seg_file_separator=self._separator)

    def get_labels(self):
        """Returns the dict of labels"""
        return self._label_dict
