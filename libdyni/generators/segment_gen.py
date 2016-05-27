import logging
from libdyni
from libdyni.generators.audio_frame_gen import AudioFrameGen
from libdyni.features.energy_extractor import EnergyExtractor
from libdyni.features.spectral_flatness import SpectralFlatnessExtractor
from libdyni.features.mel_spectrum_extractor import MelSpectrumExtractor
from libdyni.features.frame_feature_processor import FrameFeatureProcessor
from libdyni.features.segment_feature_processor import SegmentFeatureProcessor
from libdyni.features.audio_chunk_extractor import AudioChunkExtractor
from libdyni.features.activity_detection import ActivityDetection
from libdyni.utils.segment_container import create_segment_containers_from_audio_files


logger = logging.getLogger(__name__)



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





    def execute(self):

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


