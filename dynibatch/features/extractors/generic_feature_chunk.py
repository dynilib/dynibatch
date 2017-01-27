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
from os.path import join, splitext
import joblib
from dynibatch.features.extractors.segment_feature import SegmentFeatureExtractor


logger = logging.getLogger(__name__)


class GenericChunkExtractor(SegmentFeatureExtractor):
    """Extract chunks of features.

    This extractor can be used to get chunks of features having any sample rate
    (as opposed to the FrameFeatureChunkExtractor specifically designed for
    frame-based features, as defined in the feature container).
    """

    def __init__(self, name, sample_rate, generic_feature_root,
            extension=".pkl", pca=None, scaler=None):
        """Initializes frame feature chunk extractor.

        Args:
            name (str): name of the feature.
            sample_rate (int): sample rate of the feature
            generic_feature_root (str): root path of the features (every audio
                file must have a corrsponding feature file, with the same path, e.g.
                <audio_root>/some/path/somefile.wav ->
                <generic_feature_root>/some/path/somefile.<extension>)
            pca (sklearn.decomposition.PCA) (optional): Principal component analysis
                (PCA)
            scaler (sklearn.preprocessing.StandardScaler) (optional): standardize
                features by removing the mean and scaling to unit variance.

        Note: if both scaler and pca are set, the pca is performed first
        """

        super().__init__()
        self._name = name
        self._sample_rate = sample_rate
        self._generic_feature_root = generic_feature_root
        self._pca = pca
        self._scaler = scaler

    @property
    def name(self):
        return self._name

    def execute(self, segment_container):

        features = joblib.load(join(self._generic_feature_root, splitext(
            segment_container.audio_path)[0] + ".pkl")).T

        for s in segment_container.segments:

            start_ind = int(s.start_time * self._sample_rate)
            end_ind = int(start_ind + s.duration * self._sample_rate)

            # If features might not be computed over the whole file,
            # not all segments will have data.
            if end_ind >= features.shape[0]:
                break

            data = features[start_ind:end_ind]
            if self._pca:
                data = self._pca.transform(data)
            if self._scaler:
                data = self._scaler.transform(data)
            s.features[self.name] = data
