import logging
from os.path import join, splitext
from sklearn.externals import joblib
from libdyni.features.segment_feature_extractor import SegmentFeatureExtractor


logger = logging.getLogger(__name__)


class ChirpletsChunkExtractor(SegmentFeatureExtractor):
    """Ugly chirplets-specific chunk extractor.
    Waiting for chirplets code to be integrated to libdyni.
    """

    def __init__(self, sample_rate, chirplets_root, scaler):
        super().__init__()
        self._sample_rate = sample_rate
        self._chirplets_root = chirplets_root
        self._scaler = scaler

    @property
    def name(self):
        return "chirplets"

    def time_to_frame_ind(self, time):
        """Hardcoded as long as chirplets code is not integrated into libdyni"""
        return int(self._sample_rate * time / 100.)

    def execute(self, segment_container):

        chirplets = joblib.load(join(self._chirplets_root,
                splitext(segment_container.audio_path)[0] + ".0.jl"))

        for s in segment_container.segments:
            
            start_ind = self.time_to_frame_ind(s.start_time)
            end_ind =  start_ind + self.time_to_frame_ind(s.duration)

            # chirplets are not computed over the whole file (only over the greatest power
            # of 2 smaller than file size), so not all segments will have data
            if not end_ind < chirplets.shape[1]:
                break

            if self._scaler:
                s.features[self.name] =  (chirplets[:, start_ind:end_ind].T - self._scaler.mean_[0]) / self._scaler.scale_[0]
            else:
                s.features[self.name] =  chirplets[:, start_ind:end_ind].T
