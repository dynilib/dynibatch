import logging
from libdyni.utils.segment_container import \
    create_segment_containers_from_audio_files
from libdyni.utils.exceptions import GeneratorError


LOGGER = logging.getLogger(__name__)


class SegmentContainerGenerator:
    """Segment container generator
    Segments contain labels and all required features
    """

    def __init__(self,
                 audio_root,
                 segment_feature_processor,
                 label_parser=None,
                 dataset=None,
                 seg_duration=0.5,
                 seg_overlap=0.9,
                 is_random_order=False,
                 is_stratify=False):

        self._audio_root = audio_root
        self._label_parser = label_parser
        self._sf_pro = segment_feature_processor
        self._dataset = dataset
        self._seg_duration = seg_duration
        self._seg_overlap = seg_overlap
        self._is_random_order = is_random_order
        if is_stratify:
            self._stratification = self._label_parser
        else:
            self._stratification = None

        self._sc_gen = None

    def start(self):
        # create segment container with fixed-length segments
        self._sc_gen = create_segment_containers_from_audio_files(
            self._audio_root,
            self._is_random_order,
            label_parser=self._stratification,
            seg_duration=self._seg_duration,
            seg_overlap=self._seg_overlap)

    def reset(self):
        self.start()

    def execute(self):
        if self._sc_gen is None:
            raise GeneratorError("Generator is not started")

        # process
        for sc in self._sc_gen:

            if self._dataset and sc.audio_path not in self._dataset:
                continue

            if self._label_parser:
                label = self._label_parser.get_label(sc.audio_path)
                sc.labels = label

            # extract features
            self._sf_pro.execute(sc)

            yield sc

        self._sc_gen = None
