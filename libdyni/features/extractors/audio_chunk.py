import logging
from os.path import join
from soundfile import SoundFile
import numpy as np
from libdyni.features.extractors.segment_feature import SegmentFeatureExtractor


LOGGER = logging.getLogger(__name__)


class AudioChunkExtractor(SegmentFeatureExtractor):

    def __init__(self, audio_root, sample_rate):
        super().__init__()
        self._audio_root = audio_root
        self._sample_rate = sample_rate

    @property
    def name(self):
        """
        Returns:
            The name of SegmentFrameBasedFeatureExtractor, it is also its type
        """
        return "audio_chunk"

    def execute(self, segment_container):

        with SoundFile(join(self._audio_root,
                            segment_container.audio_path)) as f:

            if not f.seekable():
                raise ValueError("file must be seekable")

            for s in segment_container.segments:

                start_time = s.start_time
                end_time = s.end_time

                n_samples = int(np.rint(
                    (end_time - start_time) * self._sample_rate))

                start_ind = int(start_time * self._sample_rate)

                if start_ind + n_samples > len(f):
                    # TODO (jul) use specific exception or, maybe better, just remove it
                    # because an exception might already be raised in f.read below
                    raise ValueError(
                        "Segments {0}-{1} exceeds file {2} duration".format(
                            start_time, end_time, segment_container.audio_path))

                f.seek(start_ind)
                s.features[self.name] = f.read(n_samples, dtype="float32")
