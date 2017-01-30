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


from collections import defaultdict
from enum import Enum
import joblib

from dynibatch.utils.exceptions import ParameterError


class CommonLabels(Enum):
    no_activity = -1
    garbage = -2
    unknown = -3


class Segment:
    """
    Segments are the base elements to be fed to the learning algorithm: 1
    segment = 1 observation.

    Audio files are split into overlapping fixed-length segments, stored in
    segment containers.

    Every segment, along with its parent segment container, contains all the
    data needed to feed a mini-batch (label, features, whether it contains
            activity or not).
    """

    def __init__(self, start_time, end_time, label=CommonLabels.unknown.value):
        """Initializes segment.

        Args:
            start_time (float): start time in second
            end_time (float): end time in second
            label (int): label index
        """

        if start_time < 0 or end_time <= start_time:
            raise ParameterError(
                "Wrong time parameters: start_time must be" +
                "greater than 0 and end_time must be greater than start_time")

        self._start_time = start_time
        self._end_time = end_time
        self._label = label
        self._features = dict()

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time

    @property
    def duration(self):
        return self._end_time - self._start_time

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    @property
    def features(self):
        return self._features

    @staticmethod
    def load(path):
        segment = joblib.load(path)
        if not isinstance(segment, Segment):
            raise ParameterError(
                "Object in {} is not an instance of Segment".format(path))
        return segment


def set_segment_labels(segments_from, segments_to, overlap_ratio=0.5):
    """
    Map label from segments_from to segments_to such as a given segment s
    in segments_to has the label with more overlap in segment_from, only if this
    overlap is >= (s duration) * overlap_ratio.

    Args:
        segments_from (Segment iterator): Segment objects from which the
            labels are extracted
        segments_to (Segment iterator): Segment objects to which the labels are
            set
        overlap_ratio (float in [0, 1]): min overlap ratio
    """

    for s_to in segments_to:
        labels = defaultdict(float)
        for s_from in segments_from:
            overlap = _get_overlap(s_from.start_time,
                                   s_from.end_time,
                                   s_to.start_time,
                                   s_to.end_time)
            labels[s_from.label] += overlap

        if labels:

            # get key and value for max value
            # TODO manage several labels with max values
            k = max(labels, key=labels.get)
            max_value = labels[k]

            # check overlap ratio
            if max_value >= (s_to.end_time-s_to.start_time) * overlap_ratio:
                s_to.label = k
                continue

        s_to.label = CommonLabels.unknown.value


def _get_overlap(start1, end1, start2, end2):
    """Get overlap between the intervals [start1, end1] and [start2, end2].

    Args:
        start1 (float)
        end1 (float)
        start2 (float)
        end2 (float)
    """
    return max(0, min(end1, end2) - max(start1, start2))
