import logging
from libdyni.utils.segment_container import \
    create_segment_containers_from_audio_files
from libdyni.utils.segment import set_segment_labels
from libdyni.parsers.label_parsers import FileLabelParser, SegmentLabelParser
from libdyni.utils.exceptions import GeneratorError


logger = logging.getLogger(__name__)


class SegmentContainerGenerator:
    """Segment container generator

    The segment container generator yields SegmentContainer objects with
    segments containing all the features specified in segment_feature_processor
    and labels as specified by label_parser.
    """

    def __init__(self,
                 audio_root,
                 segment_feature_processor,
                 label_parser=None,
                 dataset=None,
                 seg_duration=0.5,
                 seg_overlap=0.9,
                 randomize=False,
                 stratify=False):
        """Initializes segment container generator.

        Args:
            audio_root (str): rooth path to the audio files
            segment_feature_processor (SegmentFeatureProcessor)
            label_parser (FileLabelParser or SegmentLabelParser)
            dataset (list of str): list of audio files to process. If set to None,
                all the audio files in audio_root are processed.
            seg_duration (float): segment duration in seconds
            seg_overlap (float): segment overlap ratio. Segments overlap by
                seg_duration * seg_overlap seconds
            randomize (bool): randomize the SegmentContainer objects
            stratify (bool): TOREMOVE
        """

        self._audio_root = audio_root
        self._label_parser = label_parser
        self._sf_pro = segment_feature_processor
        self._dataset = dataset
        self._seg_duration = seg_duration
        self._seg_overlap = seg_overlap
        self._randomize = randomize
        if stratify:
            self._stratification = self._label_parser
        else:
            self._stratification = None

        self._sc_gen = None

    def start(self):
        
        # create segment container with fixed-length segments
        self._sc_gen = create_segment_containers_from_audio_files(
            self._audio_root,
            self._randomize,
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
                if isinstance(self._label_parser, FileLabelParser):
                    label = self._label_parser.get_label(sc.audio_path)
                    sc.labels = label
                elif isinstance(self._label_parser, SegmentLabelParser):
                    sc_from = self._label_parser.get_segment_container(sc.audio_path)
                    set_segment_labels(sc_from.segments, sc.segments)

            # extract features
            self._sf_pro.execute(sc)

            yield sc

        self._sc_gen = None
