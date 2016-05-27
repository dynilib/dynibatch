from collections import defaultdict
import joblib


class common_labels:
    no_activity = "na"
    garbage = "ga"
    unknown = "un"


class Segment:
    
    def __init__(self, start_time, end_time, label=common_labels.unknown):
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
            raise Exception("Object in {} is not an instance of Segment".format(path))
        return s


def set_segment_labels(segments_from, segments_to, overlap_ratio=0.5):
    """
    Set label of segments in segments_to to the label with more overlap.
    Args::
        - overlap_ratio: min overlap ratio, in [0, 1]
    """

    for s_to in segments_to:
        _labels = defaultdict(float)
        for s_from in segments_from:
            overlap = _get_overlap(s_from.start_time, s_from.end_time, s_to.start_time, s_to.end_time)
            _labels[s_from.label] += overlap

        if _labels:

            # get key and value for max value
            k = max(_labels, key=_labels.get) # TODO manage several labels with max values
            v = _labels[k]

            # check overlap ratio
            if v >= (s_to.end_time-s_to.start_time) * overlap_ratio:
                s_to.label = k
                continue

        s_to.label = labels.unknown


def _get_overlap(start1, end1, start2, end2):
    """
    Get overlap between the intervals [start1, end1] and [start2, end2]
    """
    return max(0, min(end1, end2) - max(start1, start2))

