# -*- coding: utf-8 -*-
import os
import logging
import numpy as np
from itertools import compress
import joblib
from libdyni.utils.feature_container import FC_EXTENSION, FeatureContainer
from libdyni.features.frame_feature_extractor import FrameFeatureExtractor
from libdyni.utils.audio import info
from libdyni.features.frame_feature_extractor import *


logger = logging.getLogger(__name__)


class FrameFeatureGen:

    def __init__(self,
            audio_frame_gen,
            feature_extractors,
            feature_container_root=None):

        # TODO use Python abc (Abstract Base Classes)?
        if not all(isinstance(fe, FrameFeatureExtractor) for fe in feature_extractors):
            raise Exception("All feature extractors must be instances of FrameFeatureExtractor.")

        self._feature_container_root = feature_container_root
        self.__audio_frame_gen = audio_frame_gen
        self.__feature_extractors = feature_extractors

    @property
    def feature_container_root(self):
        return self._feature_container_root

    def execute(self, audio_path):
        """
        Args:
            audio_path: tuple (audio root, audio path relative to audio root)
        Returns tuple (fc, created)
        """

        if not type(audio_path) == tuple:
            raise Exception("The first argument must be a tuple (<audio path root>, <audio relative path>)")

        fc = None
        has_features = [False for fe in self.__feature_extractors]

        # check if feature container exists and has all required features
        # move to FeatureContainer
        # TODO create some real cache functions (check https://github.com/dnouri/nolearn/blob/master/nolearn/cache.py)
        if self._feature_container_root:
            feature_container_path = os.path.join(
                    self._feature_container_root,
                    os.path.splitext(os.path.basename(audio_path[1]))[0] + FC_EXTENSION)
            fc = FeatureContainer.load(feature_container_path)
            if fc:
                has_features = fc.has_features([(fe.name, fe.config) for fe in self.__feature_extractors])
                if all(has_features):
                    logger.debug("Feature container {} with all required features found!".format(
                        feature_container_path))
                    return fc, False

        # get audio file info
        # TODO move to audio_frame_gen module?
        _info = info(os.path.join(*audio_path))

        # number of samples (info.frames is the number of samples per channel)
        n_samples = int((_info.frames -  self.__audio_frame_gen.win_size + self.__audio_frame_gen.hop_size) / self.__audio_frame_gen.hop_size)

        if not fc or not any(has_features):
            # if fc has none of the desired features, create a new one
            fc = FeatureContainer(audio_path[1],
                    _info.samplerate,
                    self.__audio_frame_gen.win_size,
                    self.__audio_frame_gen.hop_size)


        compute_spectrum = False
        compute_power_spectrum = False

        for fe in self.__feature_extractors:
            # allocate memory for features
            # TODO move to FeatureContainer constructor?
            fc.features[fe.name]["data"] = np.empty((n_frames, fe.size), dtype="float32")
            fc.features[fe.name]["config"] = fe.config
            
            # check what to compute
            if isinstance(fe, SpectrumFrameFeatureExtractor):
                compute_spectrum = True
            elif isinstance(fe, PowerSpectrumFrameFeatureExtractor):
                compute_spectrum = True
                compute_power_spectrum = True

        frame_gen = self.__audio_frame_gen.execute(os.path.join(*audio_path))
        for i, frame in enumerate(frame_gen):

            if compute_spectrum:
                spectrum = np.abs(np.fft.rfft(frame))
            if compute_power_spectrum:
                power_spectrum = spectrum ** 2

            # TODO run every feature extractor in a different process
            for fe in compress(self.__feature_extractors, [not hf for hf in has_features]):
                if isinstance(fe, AudioFrameFeatureExtractor):
                    fc.features[fe.name]["data"][i] = fe.execute(frame)
                elif isinstance(fe, SpectrumFrameFeatureExtractor):
                    fc.features[fe.name]["data"][i] = fe.execute(spectrum)
                elif isinstance(fe, PowerSpectrumFrameFeatureExtractor):
                    fc.features[fe.name]["data"][i] = fe.execute(power_spectrum)

        # if feature_container_root is specified, write feature container
        if self._feature_container_root:
            fc.save(self._feature_container_root)

        return fc, True


