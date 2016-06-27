import pytest
import os

import numpy as np
import soundfile as sf

from libdyni.generators.audio_frame_gen import AudioFrameGen
from libdyni.generators.audio_frame_gen import Window
from libdyni.generators.segment_container_gen import SegmentContainerGenerator
from libdyni.parsers.label_parsers import CSVLabelParser
from libdyni.features.segment_feature_processor import SegmentFeatureProcessor
from libdyni.features.extractors.audio_chunk import AudioChunkExtractor

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

TEST_AUDIO_PATH_TUPLE_1 = (DATA_PATH, "ID0132.wav")
TEST_AUDIO_PATH_TUPLE_2 = (DATA_PATH, "ID1238.wav")
TEST_CSVLABEL_PATH = os.path.join(DATA_PATH, "labels.csv")


class TestAudioFrameGen:

    def test_init(self):
        try:
            win_size = 256
            hop_size = 128
            AudioFrameGen(win_size, hop_size)
        except Exception as e:
            pytest.fail("Unexpected Error: {}".format(e))

    def test_execute(self):
            win_size = 256
            hop_size = 128
            af_gen = AudioFrameGen(win_size, hop_size, win_type=Window.rect)
            af_gen_e = af_gen.execute(os.path.join(*TEST_AUDIO_PATH_TUPLE_1))
            next(af_gen_e) # 1st frame
            frame = next(af_gen_e) # 2nd frame
            data, sample_rate = sf.read(os.path.join(*TEST_AUDIO_PATH_TUPLE_1))
            assert np.all(data[128:128+256] == frame)


class TestSegmentContainerGenerator:
    
    @pytest.fixture(scope="module")
    def ac_ext(self):
        sample_rate = 22050
        return AudioChunkExtractor(TEST_AUDIO_PATH_TUPLE_1[0], sample_rate)

    def test_init(self):
        try:
            parser = CSVLabelParser(TEST_CSVLABEL_PATH)
            sf_pro = SegmentFeatureProcessor([])
            SegmentContainerGenerator(
                    "fake_audio_root",
                    sf_pro,
                    label_parser=parser)
        except Exception as e:
            pytest.fail("Unexpected Error: {}".format(e))

    def test_start(self):
        try:
            parser = CSVLabelParser(TEST_CSVLABEL_PATH)
            sf_pro = SegmentFeatureProcessor([])
            sc_gen = SegmentContainerGenerator(
                    "fake_audio_root",
                    sf_pro,
                    label_parser=parser)
            sc_gen.start()
        except Exception as e:
            pytest.fail("Unexpected Error: {}".format(e))

    def test_execute(self, ac_ext):
        
        sample_rate = 22050
        seg_duration = 0.1
        seg_overlap = 0.5

        parser = CSVLabelParser(TEST_CSVLABEL_PATH)
        sf_pro = SegmentFeatureProcessor([ac_ext])
        sc_gen = SegmentContainerGenerator(
                TEST_AUDIO_PATH_TUPLE_1[0],
                sf_pro,
                label_parser=parser,
                seg_duration=seg_duration,
                seg_overlap=seg_overlap)
        sc_gen.start()
        sc_list = [sc for sc in sc_gen.execute()]
        
        id0132_data, sr = sf.read(os.path.join(*TEST_AUDIO_PATH_TUPLE_1))
        id1238_data, sr = sf.read(os.path.join(*TEST_AUDIO_PATH_TUPLE_2))
        
        first_seg = sc_list[0].segments[0]
        first_seg_ref = id0132_data[:int(seg_duration*sample_rate)]
        last_seg = sc_list[1].segments[-1]
        start_time = 0.0
        while start_time + seg_duration < len(id1238_data) / sample_rate:
            start_time += seg_duration * seg_overlap
        start_time -= seg_duration * seg_overlap
        last_seg_ref = \
            id1238_data[int(start_time*sample_rate):int((start_time+seg_duration)*sample_rate)]
        
        assert (len(sc_list) == 2 and
                np.all(first_seg_ref == first_seg.features["audio_chunk"]) and
                np.all(last_seg_ref == last_seg.features["audio_chunk"]))
                



