import os
from collections import defaultdict
import joblib


FC_EXTENSION = ".fc.jl"


class FeatureContainer:
    def __init__(self, audio_path, sample_rate, win_size, hop_size):

        self.audio_path = audio_path  # relative to some root data path
        self.sample_rate = sample_rate
        self.win_size = win_size
        self.hop_size = hop_size
        self.features = defaultdict(dict)

    def has_features(self, features):
        """Check whether the feature container has a set of features with a
        given config.
        Args:
            features: list of tuple (name, config)
        Returns a list of booleans"""
        return [name in self.features and
                self.features.get(name).get("data") is not None and
                config == self.features.get(name).get("config", dict())
                for name, config in features]

    @staticmethod
    def load(path):
        try:
            fc = joblib.load(path)
            if not isinstance(fc, FeatureContainer):
                raise Exception(
                    "Object in {} is not an instance of FeatureContainer".format(
                        path))
            return fc
        except FileNotFoundError:
            return None

    def save(self, path, compress=0):
        joblib.dump(self,
                    os.path.join(path,
                                 os.path.splitext(
                                     os.path.basename(
                                         self.audio_path))[0] + FC_EXTENSION),
                    compress=compress)

    def time_to_frame_ind(self, time):
        return int(time * self.sample_rate / self.hop_size)
