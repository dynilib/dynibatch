import os
from collections import defaultdict
import joblib

FC_EXTENSION = ".fc.jl"


class FeatureContainer:
    def __init__(self, audio_path, sample_rate, win_size, hop_size):

        self._audio_path = audio_path  # relative to some root data path
        self._sample_rate = sample_rate
        self._win_size = win_size
        self._hop_size = hop_size
        self._features = defaultdict(dict)

    def has_features(self, features):
        """Check whether the feature container has a set of features with a
        given config.
        Args:
            features: list of tuple (name, config)
        Returns a list of booleans"""
        return [name in self._features and config ==
                self._features.get(name).get("config",
                                             dict()) for name, config in features]

    @property
    def audio_path(self):
        return self._audio_path

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def win_size(self):
        return self._win_size

    @property
    def hop_size(self):
        return self._hop_size

    @property
    def features(self):
        return self._features

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
            # TODO: better manage the exception where the method is called?
            return None

    def save(self, path, compress=0):
        joblib.dump(self,
                    os.path.join(path,
                                 os.path.splitext(
                                     os.path.basename(
                                         self._audio_path))[0] + FC_EXTENSION),
                    compress=compress)

    def time_to_frame_ind(self, time):
        return int(time * self._sample_rate / self._hop_size)
