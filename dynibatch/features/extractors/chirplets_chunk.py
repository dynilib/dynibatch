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


class ChirpletsChunkExtractor(SegmentFeatureExtractor):
    """Ugly chirplets-specific chunk extractor.
    Waiting for chirplets code to be integrated to dynibatch.

    Attributes:
        sample_rate (int): sample rate in Hz
        chirplets_root (str): path to chirplets root
        pca (sklearn.decomposition.PCA) (optional): Principal component analysis
        (PCA)
        scaler (sklearn.preprocessing.StandardScaler) (optional): standardize
            features by removing the mean and scaling to unit variance.
    Note: if both scaler and pca are set, the pca is performed first
    """

    def __init__(self, sample_rate, chirplets_root, pca, scaler):
        super().__init__()
        self._sample_rate = sample_rate
        self._chirplets_root = chirplets_root
        self._pca = pca
        self._scaler = scaler

    @property
    def name(self):
        """
        Returns:
            The name of SegmentFrameBasedFeatureExtractor, it is also its type
        """
        return "chirplets"

    def time_to_frame_ind(self, time):
        """Hardcoded as long as chirplets code is not integrated into dynibatch"""
        return int(self._sample_rate * time / 100.)

    def execute(self, segment_container):

        chirplets = joblib.load(join(self._chirplets_root, splitext(
            segment_container.audio_path)[0] + ".pkl")).T

        for s in segment_container.segments:

            start_ind = self.time_to_frame_ind(s.start_time)
            end_ind = start_ind + self.time_to_frame_ind(s.duration)

            # chirplets are not computed over the whole file (only over the greatest power
            # of 2 smaller than file size), so not all segments will have data
            if end_ind >= chirplets.shape[0]:
                break

            data = chirplets[start_ind:end_ind]
            if self._pca:
                data = self._pca.transform(data)
            if self._scaler:
                data = self._scaler.transform(data)
            s.features[self.name] = data
