import pytest
import os

import numpy as np
import soundfile as sf

from libdyni.generators.audio_frame_gen import AudioFrameGen
from libdyni.generators.audio_frame_gen import Window

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

TEST_AUDIO_PATH_TUPLE = (DATA_PATH, "ID0132.wav")


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
            af_gen_e = af_gen.execute(os.path.join(*TEST_AUDIO_PATH_TUPLE))
            next(af_gen_e) # 1st frame
            frame = next(af_gen_e) # 2nd frame
            data, sample_rate = sf.read(os.path.join(*TEST_AUDIO_PATH_TUPLE))
            assert np.all(data[128:128+256] == frame)

