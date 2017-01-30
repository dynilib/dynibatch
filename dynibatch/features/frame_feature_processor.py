#The MIT License
#
#Copyright (c) 2017 DYNI machine learning & bioacoustics team - Univ. Toulon
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of
#this software and associated documentation files (the "Software"), to deal in
#the Software without restriction, including without limitation the rights to
#use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
#the Software, and to permit persons to whom the Software is furnished to do so,
#subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
#FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
#IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
#CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import logging
import os
from itertools import compress

import numpy as np

from dynibatch.features.extractors import frame_feature as ffe
from dynibatch.utils import audio
from dynibatch.utils.feature_container import FeatureContainer, FC_EXTENSION
from dynibatch.utils import exceptions

__all__ = ['FrameFeatureProcessor']

logger = logging.getLogger(__name__)


class FrameFeatureProcessor(object):
    # TODO (jul) change name to FrameFeatureExtractor and replace current
    # FrameFeatureExtractor by something else.
    """Class holding all objects needed to run frame-based feature extractors.

    If feature_container_root is set, an existing feature container with the
    requested features is searched for before executing the feature extractors.
    If it is not found, a new one is created and written.
    """

    def __init__(self,
                 audio_frame_gen,
                 feature_extractors,
                 feature_container_root=None):
        """Initializes frame feature processor.

        Args:
            audio_frame_gen (AudioFrameGen): object yielding audio frame to feed the
                feature extractors.
            feature_extractors (list of FrameFeatureExtractor): list of feature
                extractors to be executed.
            feature_container_root (str) (optional): path where the feature
                containers are loaded/saved (some kind of cache).
        """

        # TODO (jul) use Python abc (Abstract Base Classes)?
        if not all(isinstance(fe, ffe.FrameFeatureExtractor)
                   for fe in feature_extractors):
            raise TypeError('All feature extractors must be instances of ' +
                            'FrameFeatureExtractor.')

        # TODO (jul): convert some attributes to properties to make them
        # immutable?
        self._audio_frame_gen = audio_frame_gen
        self._feature_extractors = feature_extractors
        self._feature_container_root = feature_container_root

    def execute(self, audio_path):
        """ Executes the feature extractors.

        Args:
            audio_path (tuple of str): path of the audio file to process, as a
            tuple (audio root, audio path relative to audio root).
            Example: ('/some/path', 'somefile.wav')

        Returns:
            A tuple (Featurecontainer, boolean) containing, respectively, the
            feature container and whether it has been created or not.
        """

        if not isinstance(audio_path, tuple):
            raise exceptions.ParameterError('The first argument must be a tuple' +
                                            '<audio path root>, <audio relative path>)')

        feature_container = None
        has_features = [False for fe in self._feature_extractors]

        # check if feature container exists and has all required features
        # TODO (jul) move to FeatureContainer?
        # TODO (jul) create some real cache functions
        # (check https://github.com/dnouri/nolearn/blob/master/nolearn/cache.py)
        if self._feature_container_root:
            feature_container_path = os.path.join(
                self._feature_container_root,
                os.path.splitext(audio_path[1])[0] + FC_EXTENSION)
            feature_container = FeatureContainer.load(feature_container_path)
            if feature_container:
                has_features = feature_container.has_features([(fe.name, fe.config) \
                        for fe in self._feature_extractors])
                if all(has_features):
                    logger.debug('Feature container %s with all required features found!',
                                 feature_container_path)
                    return feature_container, False

        # TODO (jul) move to audio_frame_gen module?
        info = audio.info(os.path.join(*audio_path))
        n_samples = int((info.frames - self._audio_frame_gen._win_size) /
                        self._audio_frame_gen._hop_size) + 1

        if not feature_container or not any(has_features):
            # if fc has none of the desired features, create a new one
            feature_container = FeatureContainer(
                audio_path[1],
                info.samplerate,
                self._audio_frame_gen._win_size,
                self._audio_frame_gen._hop_size)

        compute_spectrum = False
        compute_power_spectrum = False

        for feature_extractor in self._feature_extractors:
            # allocate memory for features
            # TODO move to FeatureContainer constructor?
            feature_container.features[feature_extractor.name]["data"] = \
                np.empty((n_samples, feature_extractor.size),
                         dtype="float32")
            feature_container.features[feature_extractor.name]["config"] = feature_extractor.config

            # check what to compute
            if isinstance(feature_extractor, ffe.SpectrumFrameFeatureExtractor):
                compute_spectrum = True
            elif isinstance(feature_extractor, ffe.PowerSpectrumFrameFeatureExtractor):
                compute_spectrum = True
                compute_power_spectrum = True

        frame_gen = self._audio_frame_gen.execute(os.path.join(*audio_path))
        for i, frame in enumerate(frame_gen):
            if compute_spectrum:
                spectrum = np.abs(np.fft.rfft(frame))
            if compute_power_spectrum:
                power_spectrum = spectrum ** 2

            # TODO (jul) run every feature extractor in a different process
            for feature_extractor in compress(
                    self._feature_extractors, [not hf for hf in has_features]):
                if isinstance(feature_extractor, ffe.AudioFrameFeatureExtractor):
                    feature_container.features[feature_extractor.name]["data"][i] = \
                        feature_extractor.execute(frame)
                elif isinstance(feature_extractor, ffe.SpectrumFrameFeatureExtractor):
                    feature_container.features[feature_extractor.name]["data"][i] = \
                        feature_extractor.execute(spectrum)
                elif isinstance(feature_extractor, ffe.PowerSpectrumFrameFeatureExtractor):
                    feature_container.features[feature_extractor.name]["data"][i] = \
                        feature_extractor.execute(power_spectrum)

        # if feature_container_root is set, write feature container
        if self._feature_container_root:
            feature_container.save(self._feature_container_root)

        return feature_container, True
