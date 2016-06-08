import pytest

import numpy as np

from libdyni.features.extractors.activity_detection import ActivityDetection
from libdyni.utils.feature_container import FeatureContainer
from libdyni.utils.segment import Segment
from libdyni.utils.segment_container import SegmentContainer
from libdyni.features.extractors.audio_chunk import AudioChunkExtractor


class TestActivityDetectorExtractors:

    def test_init(self):
        try:
            ActivityDetection(
                    energy_threshold=0.3,
                    spectral_flatness_threshold=0.2)
        except Exception as e:
            pytest.fail("Unexpected Error: {}".format(e))

    def test_execute(self):

        sample_rate = 22050
        win_size = 256
        hop_size = 128

        fc = FeatureContainer(
                "fake_audio_path",
                sample_rate,
                win_size,
                hop_size)
        fc.features["energy"]["data"] = np.array([1, 1, 0.2, 0.1])
        fc.features["spectral_flatness"]["data"] = np.array([0.1, 0.5, 0.12, 0.3])

        segment_list = []
        for i in range(4):
            segment_list.append(
                    Segment(
                        (i * hop_size) / sample_rate,
                        ((i * hop_size) + win_size - 1) / sample_rate))

        sc = SegmentContainer("fake_audio_path")
        sc.segments = segment_list

        act_det = ActivityDetection(
                energy_threshold=0.3,
                spectral_flatness_threshold=0.2)

        act_det.execute(sc, fc)

        assert( sc.segments[0].activity and not sc.segments[1].activity and
                not sc.segments[2].activity and not sc.segments[3].activity)


class TestAudioChunkExtractor:

    def test_init(self):
        try:
            audio_root = "fake_audio_root"
            sample_rate = 44100
            AudioChunkExtractor(audio_root, sample_rate)
        except Exception as e:
            pytest.fail("Unexpected Error: {}".format(e))
