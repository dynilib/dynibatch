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


from os.path import basename, splitext
from dynibatch.utils import segment, segment_container


class FileLabelParser:
    pass


class SegmentLabelParser:
    pass


class CSVFileLabelParser(FileLabelParser):
    """File-based label file parser (1 audio file = 1 label).

    This object parses a CSV file (aka "file2label" file) written in the following format:
    
            <file_path><separator><label>
            <file_path><separator><label>
            <file_path><separator><label>
            ...

    where <file_path> is the path, relative to some root path, of an audio file
    (see test/data/file2label.csv for an example).

    An optional label_file argument can be set to constrain the set of labels to
    be used. This file is a simple list of label, i.e.:

            <label>
            <label>
            <label>
            ...

    If this argument is set, all files with a label which is not in the list specified in
    label_file will have their label set to segment.CommonLabels.unknown.value.
    """

    def __init__(self, *file2label_files, separator=",", label_file=None):
        """Create a label list and a file2label dict to quickly get the label from an audio
        filename.

        Args:
            file2label_files (*str): one or several file2label file.
            separator (str): character used as a separator in the
                file2label file
            label_file (str): file containing the list of labels to be
            used.
         """

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
                        self._file2label_dict[sline[0].strip()] = self._label_list.index(label) if label in self._label_list else segment.CommonLabels.unknown.value

    def get_label(self, audio_path):
        """Returns the label of audio_path

        Args:
            audio_path: (relative) audio path
        """
        return self._file2label_dict[audio_path]

    def get_labels(self):
        """Returns the list of labels"""
        return self._label_list


class CSVSegmentLabelParser(SegmentLabelParser):
    """Segment-based label file parser (1 segment = 1 label).

    This object parses CSV files (aka "seg2label" file) written in the following format:
    
            <start_time><separator><end_time><separator><label>
            <start_time><separator><end_time><separator><label>
            <start_time><separator><end_time><separator><label>
            ...

    where <start_time> and <end_time> are given in seconds (see test/data/*.seg
    for some examples).

    Every audio file must have a corresponding seg2label file in a
    seg2label_files_root. seg2label_files_root must have the same structure as
    the audio files root.

    An label_file argument must be set to specify the set of labels to
    be used. This file is a simple list of label, i.e.:

            <label>
            <label>
            <label>
            ...

    All segments with a label which is not in the list specified in label_file
    will have their label set to segment.CommonLabels.unknown.value.
    """

    def __init__(self,
            seg2label_files_root,
            label_file,
            audio_file_extension=".wav",
            seg_file_extension=".seg",
            seg_file_separator=","):
        """Create a label list.

        Args:
            seg2label_files_root (str): root path of the seg2label files.
            separator (str): character used as a separator in the
                seg2label files
            label_file: file containing the list of labels to be
                used.
            audio_file_extension (str)
            seg_file_extension (str)
            seg_file_separator (str)
         """

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
        """Returns a segment container with all the segments set to the labels
        specified in the seg2label files
        
        Args:
            audio_path

        Returns:
            SegmentContainer
        """

        seg_file_path_tuple = (self._seg2label_files_root, audio_path.replace(self._audio_file_extension, self._seg_file_extension))

        return segment_container.create_segment_container_from_seg_file(seg_file_path_tuple,
            self._label_list,
            audio_file_ext=self._audio_file_extension,
            seg_file_ext=self._seg_file_extension,
            seg_file_separator=self._seg_file_separator)
    
    def get_labels(self):
        """Returns the list of labels"""
        return self._label_list
