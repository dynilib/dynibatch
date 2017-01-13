import logging
import os
from itertools import compress

import numpy as np

from libdyni.features.extractors import frame_feature as ffe
from libdyni.utils import audio
from libdyni.utils import feature_container

__all__ = ['FrameFeatureProcessor']

logger = logging.getLogger(__name__)


class FrameFeatureProcessor(object):
    # TODO (jul) change name to FrameFeatureExtractor and replace current
    # FrameFeatureExtractor by something else.
    """Class holding all objects needed to run frame-based feature extractors.

    If feature_container_root is set, an existing feature container with the
    requested features is searched for before executing the feature extractors.
    If it is not found, a new one is created and written.

    Attributes:
        audio_frame_gen (AudioFrameGen): object yielding audio frame to feed the
            feature extractors.
        feature_extractors (list of FrameFeatureExtractor): list of feature
            extractors to be executed.
        feature_container_root (str) (optional): path where the feature
            containers are loaded/saved (some kind of cache).
    """

    def __init__(self,
                 audio_frame_gen,
                 feature_extractors,
                 feature_container_root=None):

        # TODO (jul) use Python abc (Abstract Base Classes)?
        if not all(isinstance(fe, ffe.FrameFeatureExtractor)
                   for fe in feature_extractors):
            raise TypeError('All feature extractors must be instances of ' +
                            'FrameFeatureExtractor.')

        # TODO (jul): convert some attributes to properties to make them
        # immutable?
        self.audio_frame_gen = audio_frame_gen
        self.feature_extractors = feature_extractors
        self.feature_container_root = feature_container_root

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
            raise Exception('The first argument must be a tuple' +
                            '<audio path root>, <audio relative path>)')

        fc = None
        has_features = [False for fe in self.feature_extractors]

        # check if feature container exists and has all required features
        # TODO (jul) move to FeatureContainer?
        # TODO (jul) create some real cache functions
        # (check https://github.com/dnouri/nolearn/blob/master/nolearn/cache.py)
        if self.feature_container_root:
            feature_container_path = os.path.join(
                self.feature_container_root,
                os.path.splitext(os.path.basename(audio_path[1]))[0] +
                feature_container.FC_EXTENSION)
            fc = feature_container.FeatureContainer.load(feature_container_path)
            if fc:
                has_features = fc.has_features([(fe.name, fe.config) \
                        for fe in self.feature_extractors])
                if all(has_features):
                    logger.debug('Feature container %s with all required features found!',
                                 feature_container_path)
                    return fc, False

        # TODO (jul) move to audio_frame_gen module?
        info = audio.info(os.path.join(*audio_path))
        n_samples = int((info.frames - self.audio_frame_gen.win_size) /
                        self.audio_frame_gen.hop_size) + 1

        if not fc or not any(has_features):
            # if fc has none of the desired features, create a new one
            fc = feature_container.FeatureContainer(
                audio_path[1],
                info.samplerate,
                self.audio_frame_gen.win_size,
                self.audio_frame_gen.hop_size)

        compute_spectrum = False
        compute_power_spectrum = False

        for fe in self.feature_extractors:
            # allocate memory for features
            # TODO move to FeatureContainer constructor?
            fc.features[fe.name]["data"] = np.empty((n_samples, fe.size),
                                                    dtype="float32")
            fc.features[fe.name]["config"] = fe.config

            # check what to compute
            if isinstance(fe, ffe.SpectrumFrameFeatureExtractor):
                compute_spectrum = True
            elif isinstance(fe, ffe.PowerSpectrumFrameFeatureExtractor):
                compute_spectrum = True
                compute_power_spectrum = True

        frame_gen = self.audio_frame_gen.execute(os.path.join(*audio_path))
        for i, frame in enumerate(frame_gen):
            if compute_spectrum:
                spectrum = np.abs(np.fft.rfft(frame))
            if compute_power_spectrum:
                power_spectrum = spectrum ** 2

            # TODO (jul) run every feature extractor in a different process
            # TODO (jul) convert loop to matrix computation
            for fe in compress(
                    self.feature_extractors, [not hf for hf in has_features]):
                if isinstance(fe, ffe.AudioFrameFeatureExtractor):
                    fc.features[fe.name]["data"][i] = fe.execute(frame)
                elif isinstance(fe, ffe.SpectrumFrameFeatureExtractor):
                    fc.features[fe.name]["data"][i] = fe.execute(spectrum)
                elif isinstance(fe, ffe.PowerSpectrumFrameFeatureExtractor):
                    fc.features[fe.name]["data"][i] = fe.execute(power_spectrum)

        # if feature_container_root is set, write feature container
        if self.feature_container_root:
            fc.save(self.feature_container_root)

        return fc, True
