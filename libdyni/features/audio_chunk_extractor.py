import logging
from soundfile import SoundFile
from libdyni.features.segment_feature_extractor import SegmentFeatureExtractor


logger = logging.getLogger(__name__)


class AudioChunkExtractor(SegmentFeatureExtractor):

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return "audio_chunk"
    
    def execute(self, segment_container):

        with SoundFile(segment_container.audio_path) as f:

            if not f.seekable():
                raise ValueError("file must be seekable")

            for s in segment_container:

                start_time = s.start_time
                end_time = s.end_time

                n_samples = int((end_time - start_time) * feature_container.sample_rate)

                start_ind = int(start_time * feature_container.sample_rate)
                
                if start_ind + n_samples > len(f):
                    raise Exception("Segments {0}-{1} exceeds file {2} duration".format(
                        start_time, end_time, segment_container.audio_path))

                f.seek(start_ind)
                s.features[self._name] = f.read(n_samples, dtype="float32")
