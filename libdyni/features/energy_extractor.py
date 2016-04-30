import numpy as np
from libdyni.features.frame_feature_extractor import AudioFrameFeatureExtractor


class EnergyExtractor(AudioFrameFeatureExtractor):

    def __init__(self):
        super().__init__()
    
    @property
    def name(self):
        return "energy"

    @property
    def size(self):
        return 1

    @property
    def config(self):
        return {}

    def execute(self, data):
        return np.sum(data**2)

