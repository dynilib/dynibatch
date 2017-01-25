import os
from collections import defaultdict
import joblib
from libdyni.utils import exceptions


FC_EXTENSION = ".fc.jl"


class FeatureContainer:
    def __init__(self, audio_path, sample_rate, win_size, hop_size):

        self._audio_path = audio_path  # relative to some root data path
        self._sample_rate = sample_rate
        self._win_size = win_size
        self._hop_size = hop_size
        self.features = defaultdict(dict)

    def has_features(self, features):
        """
            Check whether the feature container has a set of features with a
            given config.

            Args:
                features: list of tuple (name, config)
            Returns a list of booleans
        """

        return [name in self.features and
                self.features.get(name).get("data") is not None and
                config == self.features.get(name).get("config", dict())
                for name, config in features]

    @staticmethod
    def load(path):
        try:
            fc = joblib.load(path)
            if not isinstance(fc, FeatureContainer):
                raise exceptions.ParsingError(
                    "Object in {} is not an instance of FeatureContainer".format(
                        path))
            return fc
        except FileNotFoundError:
            return None

    def save(self, path, compress=0):
        filename =  os.path.join(path, os.path.splitext(self._audio_path)[0] + FC_EXTENSION)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        joblib.dump(self, filename, compress=compress)

    def time_to_frame_ind(self, time):
        return int(time * self._sample_rate / self._hop_size)
