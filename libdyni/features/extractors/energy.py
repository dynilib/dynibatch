import numpy as np

from libdyni.features.extractors.frame_feature import AudioFrameFeatureExtractor


class EnergyExtractor(AudioFrameFeatureExtractor):
    """Computes the energy of the signal."""

    def __init__(self):
        super().__init__()
    
    @classmethod
    def from_config_dict(cls, _, _):
        return cls()

    @property
    def name(self):
        return self.__module__.split('.')[-1]

    @property
    def size(self):
        """
            :return: size of the feature
        """
        return 1

    @property
    def config(self):
        return {}

    def execute(self, data):
        return np.sum(data**2)
