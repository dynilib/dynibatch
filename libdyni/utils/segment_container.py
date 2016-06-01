import os
import joblib
import soundfile as sf
from libdyni.utils.segment import Segment


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

    for root, dirnames, filenames in os.walk(audio_root):

        for filename in filenames:

            basename, extension = os.path.splitext(filename)
            if not extension in ALLOWED_AUDIO_EXT:
                continue  # only get audio files

            with sf.SoundFile(os.path.join(root, filename)) as f:
                sc = SegmentContainer(os.path.relpath(os.path.join(
                    root, filename), audio_root))
                n_samples = len(f)
                sample_rate = f._info.samplerate
                duration = float(n_samples) / sample_rate

                if "seg_duration" in kwargs and kwargs["seg_duration"]:
                    sc._segments = split_data(duration, **kwargs)
                else:
                    sc._segments.append(Segment(0, duration))

                yield sc


def create_segment_containers_from_seg_files(seg_file_root,
                                             audio_file_ext=".wav",
                                             seg_file_ext=".seg",
                                             seg_file_separator="\t"):
    """
    Args:
        - seg_file_root
        - (seg_file_ext)
        - audio_file_ext
    Yields: segment container
    """

    for root, dirnames, filenames in os.walk(seg_file_root):

        for filename in filenames:

            basename, ext = os.path.splitext(filename)
            if not ext == seg_file_ext:
                continue  # only get seg files

            with open(os.path.join(root, filename), "r") as f:

                sc = SegmentContainer(
                    os.path.relpath(
                        os.path.join(root,
                                     filename.replace(seg_file_ext,
                                                      audio_file_ext)),
                        seg_file_root))
                for line in f:
                    start_time, end_time, label = _parse_segment_file_line(
                        line, seg_file_separator)
                    sc._segments.append(Segment(start_time, end_time, label))

                yield sc


def load_segment_containers_from_dir(path):

    for root, dirnames, filenames in os.walk(path):

        for filename in filenames:

            basename, ext = os.path.splitext(filename)
            if not ext == SC_EXTENSION:
                continue  # only get segment containers

            yield SegmentContainer.load(os.path.join(root, filename))


def split_data(file_duration, seg_duration, seg_overlap=0.5):

    segments = []

    start_time = 0.0
    while start_time + seg_duration < file_duration:
        segments.append(Segment(start_time, start_time + seg_duration))
        start_time += (1 - seg_overlap) * seg_duration
    return segments


def _parse_segment_file_line(line, field_separator):

    # TODO (jul) raise exception when format is not right
    # replace "," by "." just in case time is set as xx,xx
    tmp = line.strip().replace(",", ".").split(field_separator)
    start_time = float(tmp[0])
    end_time = float(tmp[1])
    label = tmp[2]
    return start_time, end_time, label
