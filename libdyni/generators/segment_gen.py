import logging


LOGGER = logging.getLogger(__name__)


class SegmentGenerator:
    """Segment generator
    Segments contain all required features
    """
    # TODO randomize segment containers order

    def __init__(self,
                 audio_root,
                 feature_container_root,
                 label_parser,
                 sample_rate=22050,
                 win_size=256,
                 hop_size=128,
                 seg_duration=0.5,
                 seg_overlap=0.9,
                 energy_threshold=0.2,
                 spectral_flatness_threshold=0.1):
        self._audio_root = audio_root
        self._feature_container_root = feature_container_root
        self._label_parser = label_parser
        self._sample_rate = sample_rate
        self._win_size = win_size
        self._hop_size = hop_size
        self._seg_duration = seg_duration
        self._seg_overlap = seg_overlap
        self._energy_threshold = energy_threshold
        self._spectral_flatness_threshold = spectral_flatness_threshold

    def execute(self):
        # TODO _sc_gen and _sf_pro are not resolved
        # process
        for sc in self._sc_gen:

            # get label
            label = self._label_parser.get_label(sc.audio_path)

            # set label
            sc.labels = label

            # detect activity
            self._sf_pro.execute(sc)

            for s in sc.segments:
                yield s
