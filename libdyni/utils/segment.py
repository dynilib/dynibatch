from collections import defaultdict
import joblib

from libdyni.utils.exceptions import ParameterError


class common_labels:
    no_activity = "na"
    garbage = "ga"
    unknown = "un"


class Segment:
    def __init__(self, start_time, end_time, label=common_labels.unknown):

        if start_time < 0 or end_time <= start_time:
            raise ParameterError(
                "Wrong time parameters: start_time must be" +
                "greater than 0 and end_time must be greater than start_time")

        self._start_time = start_time
        self._end_time = end_time
        self._label = label
        self._features = dict()

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
        s = joblib.load(path)
        if not isinstance(s, Segment):
            raise ParameterError(
                "Object in {} is not an instance of Segment".format(path))
        return s


def set_segment_labels(segments_from, segments_to, overlap_ratio=0.5):
    """
    Map label from segments_from to segments_to such as a given segment s
    in segments_to has the label with more overlap in segment_from, only if this
    overlap is >= (s duration) * overlap_ratio.
    Args::
        - overlap_ratio: min overlap ratio, in [0, 1]
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
            v = labels[k]

            # check overlap ratio
            if v >= (s_to.end_time-s_to.start_time) * overlap_ratio:
                s_to.label = k
                continue

        s_to.label = common_labels.unknown


def _get_overlap(start1, end1, start2, end2):
    """
    Get overlap between the intervals [start1, end1] and [start2, end2]
    """
    return max(0, min(end1, end2) - max(start1, start2))
