import os
import re
import joblib

import soundfile as sf

from libdyni.utils.segment import Segment
from libdyni.utils import exceptions


SC_EXTENSION = ".sc.jl"
ALLOWED_AUDIO_EXT = [".wav"]


class SegmentContainer:

    def __init__(self, audio_path):
        self._audio_path = audio_path  # relative to some root data path
        self._segments = []

    @property
    def audio_path(self):
        return self._audio_path

    @property
    def segments(self):
        return self._segments

    @segments.setter
    def segments(self, segments):
        self._segments = segments

    @staticmethod
    def load(path):
        sc = joblib.load(path)
        if not isinstance(sc, SegmentContainer):
            raise TypeError(
                "Object in {} is not an instance of SegmentContainer".format(
                    path))
        return sc

    @property
    def labels(self):
        # get segment label set
        return set(s.label for s in self._segments)

    @labels.setter
    def labels(self, label):
        # set label to all segments
        # TODO: maybe create a specific method for that, not a property setter?
        for s in self._segments:
            s.label = label

    @property
    def n_segments(self):
        return len(self._segments)

    @property
    def n_active_segments(self):
        return sum(1 for s in self._segments if
                   hasattr(s, "activity") and s.activity)

    def save(self, path, compress=0):
        joblib.dump(self,
                    os.path.join(path,
                                 os.path.splitext(
                                     os.path.basename(
                                         self._audio_path))[0] + SC_EXTENSION),
                    compress=compress)

    def n_segments_with_label(self, label):
        return sum(1 for s in self._segments if s.label == label)

    def n_active_segments_with_label(self, label):
        return sum(1 for s in self._segments if
                   hasattr(s, "activity") and s.activity and s.label == label)

    def has_features(self, features):
        # assumes that if the last segment has the feature,
        # all segments do
        # return a list of boolean
        if (self._segments and
                hasattr(self._segments[-1], "features")):
            return [f in self._segments[-1].features for f in features]
        return [False for f in features]


def create_segment_containers_from_audio_files(audio_root, **kwargs):
    """
    Args:
        audio_root
        (seg_duration
        (seg_overlap)
    Yields: segment container
    """

    for root, _, filenames in os.walk(audio_root):

        # TODO: randomize (we use sorted so that the list is the always the same
        # for the unit tests)
        for filename in sorted(filenames):

            _, extension = os.path.splitext(filename)
            if not extension in ALLOWED_AUDIO_EXT:
                continue  # only get audio files

            audio_path_tuple = (
                audio_root,
                os.path.relpath(os.path.join(root, filename), audio_root))

            yield create_segment_containers_from_audio_file(
                audio_path_tuple,
                **kwargs)


def create_segment_containers_from_audio_file(audio_path_tuple, **kwargs):
    """
    Args:
        audio_path_tuple: audio file path as a tuple (<audio root>, <audio file
            relative path>)
        (seg_duration
        (seg_overlap)
    Yields: segment container
    """

    # if a str is passed as audio_path_tuple, sc.audio_path will be wrong,
    # so we must make sure it is a tuple
    if not isinstance(audio_path_tuple, tuple):
        raise TypeError("audio_path_tuple must be a tuple")

    with sf.SoundFile(os.path.join(*audio_path_tuple)) as f:
        sc = SegmentContainer(audio_path_tuple[1])
        n_samples = len(f)
        sample_rate = f._info.samplerate
        duration = float(n_samples) / sample_rate

        if "seg_duration" in kwargs and kwargs["seg_duration"]:
            sc.segments = create_fixed_duration_segments(duration, **kwargs)
        else:
            sc.segments.append(Segment(0, duration))

        return sc


def create_segment_containers_from_seg_files(seg_file_root,
                                             audio_file_ext=".wav",
                                             seg_file_ext=".seg",
                                             seg_file_separator="\t"):
    """
    Args:
        - seg_file_root
        - (audio_file_ext)
        - (seg_file_ext)
        - (seg_file_separator)
    Yields: segment container
    """

    for root, _, filenames in os.walk(seg_file_root):

        for filename in filenames:

            _, ext = os.path.splitext(filename)
            if not ext == seg_file_ext:
                continue  # only get seg files

            seg_file_path_tuple = (
                seg_file_root,
                os.path.relpath(
                    os.path.join(root, filename.replace(seg_file_ext, audio_file_ext)),
                    seg_file_root))

            yield create_segment_containers_from_seg_file(
                seg_file_path_tuple,
                audio_file_ext,
                seg_file_ext,
                seg_file_separator)


def create_segment_containers_from_seg_file(seg_file_path_tuple,
                                            audio_file_ext=".wav",
                                            seg_file_ext=".seg",
                                            seg_file_separator="\t"):
    """
    Args:
        - seg_file_path_tuple: seg file path as a tuple (<audio root>,
            <audio file relative path>)
        - (audio_file_ext)
        - (seg_file_ext)
        - (seg_file_separator)
    Yields: segment container
    """

    # if a str is passed as seg_file_path_tuple, sc.audio_path will be wrong,
    # so we must make sure it is a tuple
    if not isinstance(seg_file_path_tuple, tuple):
        raise TypeError("seg_file_path_tuple must be a tuple")

    if audio_file_ext not in ALLOWED_AUDIO_EXT:
        raise exceptions.ParameterError(
            "{} is not an allowed audio file extension")

    with open(os.path.join(*seg_file_path_tuple), "r") as f:

        sc = SegmentContainer(
            seg_file_path_tuple[1].replace(seg_file_ext,
                                           audio_file_ext))
        for line in f:
            start_time, end_time, label = _parse_segment_file_line(
                line, seg_file_separator)
            sc.segments.append(Segment(start_time, end_time, label))

        return sc


def load_segment_containers_from_dir(path):

    for root, _, filenames in os.walk(path):

        for filename in filenames:

            _, ext = os.path.splitext(filename)
            if not ext == SC_EXTENSION:
                continue  # only get segment containers

            yield SegmentContainer.load(os.path.join(root, filename))


def create_fixed_duration_segments(file_duration, seg_duration, seg_overlap=0.5):

    segments = []

    start_time = 0.0
    while start_time + seg_duration < file_duration:
        segments.append(Segment(start_time, start_time + seg_duration))
        start_time += (1 - seg_overlap) * seg_duration
    return segments


def _parse_segment_file_line(line, field_separator):

    pattern = re.compile(
        "^\s*[0-9]+\.[0-9]+\s*" + re.escape(field_separator) +
        "\s*[0-9]+\.[0-9]+\s*" + re.escape(field_separator) + "\s*.+\s*$")
    if not line.count(field_separator) == 2 and re.match(pattern, line):
        raise exceptions.ParsingError(
            "Cannot parse line '{}'".format(line))

    tmp = line.split(field_separator)
    start_time = float(tmp[0].strip())
    end_time = float(tmp[1].strip())
    label = tmp[2].strip()
    return start_time, end_time, label
