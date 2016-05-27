import logging

import numpy as np

from libdyni.features import segment_feature_extractor as sfe

__all__ = ['ActivityDetection']

logger = logging.getLogger(__name__)


class ActivityDetection(sfe.SegmentFrameBasedFeatureExtractor):
    """Bird activity detection, based on signal energy and voiceness

    The activity detection is executed on all segments of a segment container.
    Activity is detected when the mean frame-based energy of the segment is
    higher than the (95th percentile of the energy in the file X
    energy_threshold) and the energy-weighted mean of the frame-based spectral
    flatness is lower than spectral_flatness_threshold.

    It creates an attribute 'activity', set to a boolean.

    Attributes:
        energy_threshold (float): see class description above.
        spectral_flatness_threshold (float): see class description above.

    """

    def __init__(self,
            energy_threshold=0.2,
            spectral_flatness_threshold=0.3):
        super().__init__()
        self.energy_threshold = energy_threshold
        self.spectral_flatness_threshold = spectral_flatness_threshold

    @property
    def name(self):
        return 'activity_detection'

    def execute(self, segment_container, feature_container):
        """Executes the activity detection.

        Args:
            segment_container (Segmentcontainer): segment container to process.
            feature_container (Featurecontainer): feature container containing
                the features needed to perform the activity detection.

        Creates and sets an attribute 'activity' to all segments in
            segment_container.
        """

        # TODO (jul) make sure sc has required frame features

        energy95p = np.percentile(feature_container.features["energy"]["data"],
                95)
        if energy95p == 0:
            raise Exception("The file {} is silent".format( \
                    segment_container.audio_path))

        for s in segment_container.segments:

            start_ind = feature_container.time_to_frame_ind(s.start_time)
            end_ind =  start_ind + \
                    feature_container.time_to_frame_ind(s.duration)

            en = feature_container.features["energy"]["data"][start_ind:end_ind]
            sf = feature_container.features["spectral_flatness"]["data"][start_ind:end_ind]

            mean_en = np.mean(en)
            energy_ratio = mean_en / energy95p
            spectral_flatness_w_mean = \
                    np.average(sf, weights=en) if mean_en > 0 else 0.0

            if (energy_ratio > self.energy_threshold and
                    spectral_flatness_w_mean < self.spectral_flatness_threshold):
                s.activity = True
            else:
                s.activity = False
