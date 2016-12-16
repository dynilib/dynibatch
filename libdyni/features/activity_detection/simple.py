import logging

import numpy as np

from libdyni.features.extractors import segment_feature as sfe


logger = logging.getLogger(__name__)


class Simple(sfe.SegmentFrameBasedFeatureExtractor):
    """simple bird activity detection, based on signal energy and voiceness

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

        self.frame_feature_config = [
                {
                    "name": "energy",
                    "config": {}
                    },
                {
                    "name": "spectral_flatness",
                    "config": {}
                    }
                ]

        self.energy_threshold = energy_threshold
        self.spectral_flatness_threshold = spectral_flatness_threshold

    @classmethod
    def from_config_dict(cls, audio_frame_config, feature_config):
        return cls(
                feature_config["energy_threshold"],
                feature_config["spectral_flatness_threshold"])

    @property
    def name(self):
        return self.__module__.split('.')[-1]

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
            # TODO (jul) just warn
            raise Exception(
                "The file {} is silent".format(segment_container.audio_path))

        for s in segment_container.segments:

            start_ind = feature_container.time_to_frame_ind(s.start_time)
            end_ind = start_ind + feature_container.time_to_frame_ind(
                s.duration)
            
            if (end_ind > len(feature_container.features["energy"]["data"]) or
                    end_ind > len(feature_container.features["spectral_flatness"]["data"])):
                # that can happen if the end time of the latest analysis frame
                # is earlier than the end time of the segment
                logger.debug("Segment {0:.3f}-{1:.3f} from {2} end time".format(s.start_time,
                                s.end_time, segment_container.audio_path) +
                        " exceed feature container size for extractor {}.".format(self.name))
                break

            en = feature_container.features["energy"]["data"][start_ind:end_ind]
            sf = feature_container.features["spectral_flatness"]["data"][
                start_ind:end_ind]

            mean_en = np.mean(en)
            energy_ratio = mean_en / energy95p
            spectral_flatness_w_mean = np.average(sf,
                                                  weights=en) if mean_en > 0 else 0.0

            s.activity = energy_ratio > self.energy_threshold and \
                         spectral_flatness_w_mean < self.spectral_flatness_threshold
