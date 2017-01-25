from sys import float_info
import numpy as np
from libdyni.features.extractors.frame_feature import SpectrumFrameFeatureExtractor


class SpectralFlatnessExtractor(SpectrumFrameFeatureExtractor):
    """Spectral flatness extractor.

    The spectral flatness is calculated by dividing the geometric mean of the
    power spectrum by the arithmetic mean of the power spectrum.
    """

    def __init__(self):
        super().__init__()
    
    @classmethod
    def from_config_dict(cls, audio_frame_config, feature_config):
        return cls()

    @property
    def name(self):
        return self.__module__.split('.')[-1]

    @property
    def size(self):
        return 1

    @property
    def config(self):
        return {}

    def execute(self, data):
        """Computes the spectral flatness.

        Args:
            data (numpy array): power spectrum

        Returns:
            float: spectral flatness
        """

        data = np.where(data == 0, float_info.epsilon, data) # replace 0s by epsilon to avoid log(0)
        sum_mag_bins = np.sum(data)
        return np.exp(np.sum(np.log(data)) / len(data)) / (sum_mag_bins / len(data))
