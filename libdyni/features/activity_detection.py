import numpy as np
from libdyni.features.segment_feature_extractor import SegmentFrameBasedFeatureExtractor
from libdyni.utils.segment import labels


class ActivityDetection(SegmentFrameBasedFeatureExtractor):

    def __init__(self, energy_threshold=0.2, spectral_flatness_threshold=0.3):
        super().__init__()
        self._energy_threshold = energy_threshold
        self._spectral_flatness_threshold = spectral_flatness_threshold

    @property
    def energy_threshold(self):
        return self._energy_threshold

    @property
    def spectral_flatness_threshold(self):
        return self._spectral_flatness_threshold

    @property
    def name(self):
        return "activity_detection"

    def execute(self, segment_container, feature_container):
        """
        Add and set a boolean activity attribute to segments
        """

        # TODO make sure sc has required frame features

        # compute the 95th percentile of the energy
        energy95p = np.percentile(feature_container.features["energy"]["data"], 95)

        if energy95p == 0:
            raise Exception("The file {} is silent".format(segment_container.audio_path))

        for s in segment_container.segments:

            start_ind = feature_container.time_to_frame_ind(s.start_time)
            end_ind =  start_ind + feature_container.time_to_frame_ind(s.duration)

            en = feature_container.features["energy"]["data"][start_ind:end_ind]
            sf = feature_container.features["spectral_flatness"]["data"][start_ind:end_ind]

            mean_en = np.mean(en)
            energy_ratio = mean_en / energy95p
            spectral_flatness_w_mean = np.average(sf, weights=en) if mean_en > 0 else 0.0

            if (energy_ratio > self._energy_threshold and
                    spectral_flatness_w_mean < self._spectral_flatness_threshold):
                s.activity = True
            else:
                s.activity = False
