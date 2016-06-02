from sys import float_info
import numpy as np
from libdyni.features.frame_feature_extractor import SpectrumFrameFeatureExtractor


class SpectralFlatnessExtractor(SpectrumFrameFeatureExtractor):

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        """
        Returns:
            The name of SegmentFrameBasedFeatureExtractor, it is also its type
        """
        return "spectral_flatness"

    @property
    def size(self):
        return 1

    @property
    def config(self):
        return {}

    def execute(self, data):
        data = np.where(data == 0, float_info.epsilon, data) # replace 0s by epsilon to avoid log(0)
        sum_mag_bins = np.sum(data)
        return np.exp(np.sum(np.log(data)) / len(data)) / (sum_mag_bins / len(data))
