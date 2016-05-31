import logging
from libdyni.utils.segment_container import \
    create_segment_containers_from_audio_files


logger = logging.getLogger(__name__)


class SegmentContainerGenerator:
    """Segment container generator
    Segments contain labels and all required features
    """

    def __init__(self,
                 audio_root,
                 label_parser,
                 segment_feature_processor,
                 dataset=None,
                 seg_duration=0.5,
                 seg_overlap=0.9):

        self._audio_root = audio_root
        self._label_parser = label_parser
        self._sf_pro = segment_feature_processor
        self._dataset = dataset
        self._seg_duration = seg_duration
        self._seg_overlap = seg_overlap

    def start(self):
        # create segment container with fixed-length segments
        self._sc_gen = create_segment_containers_from_audio_files(
            self._audio_root,
            seg_duration=self._seg_duration,
            seg_overlap=self._seg_overlap)

    def reset(self):
        self.start()

    def execute(self):

        # process
        for sc in self._sc_gen:

            if self._dataset and not sc.audio_path in self._dataset:
                continue

            # get label
            label = self._label_parser.get_label(sc.audio_path)
            
            # set label
            sc.labels = label

            # extract features
            self._sf_pro.execute(sc)

            yield sc
