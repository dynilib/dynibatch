import pytest
import os
import copy

import numpy as np
from sklearn.preprocessing import StandardScaler
import soundfile as sf
from librosa.feature.spectral import melspectrogram
from librosa.feature.spectral import mfcc as lr_mfcc
from librosa import logamplitude

from libdyni.features.extractors.activity_detection import ActivityDetection
from libdyni.utils.feature_container import FeatureContainer
from libdyni.utils.segment import Segment
from libdyni.utils.segment_container import SegmentContainer
from libdyni.features.frame_feature_processor import FrameFeatureProcessor
from libdyni.features.segment_feature_processor import SegmentFeatureProcessor
from libdyni.features.extractors.audio_chunk import AudioChunkExtractor
from libdyni.features.extractors.energy import EnergyExtractor
from libdyni.features.extractors.frame_feature_chunk import FrameFeatureChunkExtractor
from libdyni.features.extractors.mel_spectrum import MelSpectrumExtractor
from libdyni.features.extractors.mfcc import MFCCExtractor
from libdyni.features.extractors.spectral_flatness import SpectralFlatnessExtractor
from libdyni.generators.audio_frame_gen import AudioFrameGen
from libdyni.generators.audio_frame_gen import Window

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

TEST_AUDIO_PATH_TUPLE = (DATA_PATH, "ID0132.wav")


class TestActivityDetection:

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
    
    def test_execute(self):
        data, sample_rate = sf.read(os.path.join(*TEST_AUDIO_PATH_TUPLE))
        sc = SegmentContainer(TEST_AUDIO_PATH_TUPLE[1])
        sc.segments.append(Segment(0, 0.5))
        sc.segments.append(Segment(14, 15))
        ac_ext = AudioChunkExtractor(TEST_AUDIO_PATH_TUPLE[0], sample_rate)
        ac_ext.execute(sc)
        assert (
            np.all(sc.segments[0].features["audio_chunk"] == \
                    data[int(0*sample_rate):int(0.5*sample_rate)]) and
            np.all(sc.segments[1].features["audio_chunk"] == \
                    data[int(14*sample_rate):int(15*sample_rate)]))
    
    def test_execute_valueerror(self):
        with pytest.raises(ValueError):
            data, sample_rate = sf.read(os.path.join(*TEST_AUDIO_PATH_TUPLE))
            sc = SegmentContainer(TEST_AUDIO_PATH_TUPLE[1])
            sc.segments.append(Segment(15, 16))
            ac_ext = AudioChunkExtractor(TEST_AUDIO_PATH_TUPLE[0], sample_rate)
            ac_ext.execute(sc)


class TestEnergyExtractor:

    def test_init(self):
        try:
            EnergyExtractor()
        except Exception as e:
            pytest.fail("Unexpected Error: {}".format(e))
    
    def test_execute(self):
        data = np.ones((100,)) * 2
        en_ext = EnergyExtractor()
        assert en_ext.execute(data) == 400


class TestFrameFeatureChunkExtractor:

    @pytest.fixture(scope="module")
    def feature_container(self):
        sample_rate = 22050
        win_size = 256
        hop_size = 128
        fc = FeatureContainer(
                "fake_audio_path",
                sample_rate,
                win_size,
                hop_size)
        fc.features["fake_feature"]["data"] = np.random.sample((100, 2))
        return fc
    
    @pytest.fixture(scope="module")
    def scaler(self, feature_container):
        scaler = StandardScaler()
        scaler.fit(feature_container.features["fake_feature"]["data"])
        return scaler

    def test_init(self, scaler):
        try:
            FrameFeatureChunkExtractor("fake_feature", scaler)
        except Exception as e:
            pytest.fail("Unexpected Error: {}".format(e))
    
    def test_execute(self, feature_container):
        sc = SegmentContainer("fake_audio_path")
        sc.segments.append(Segment(0, 0.05))
        sc.segments.append(Segment(0.4, 0.5))
        s0 = sc.segments[0]
        s0_start_ind = feature_container.time_to_frame_ind(s0.start_time)
        s0_end_ind = s0_start_ind + feature_container.time_to_frame_ind(s0.duration)
        s1 = sc.segments[1]
        s1_start_ind = feature_container.time_to_frame_ind(s1.start_time)
        s1_end_ind = s1_start_ind + feature_container.time_to_frame_ind(s1.duration)

        ffc_ext = FrameFeatureChunkExtractor("fake_feature")
        ffc_ext.execute(sc, feature_container)

        assert (np.all(s0.features["fake_feature"] == \
                feature_container.features["fake_feature"]["data"][s0_start_ind:s0_end_ind]) and \
                np.all(s1.features["fake_feature"] == \
                feature_container.features["fake_feature"]["data"][s1_start_ind:s1_end_ind]))
    
    def test_execute_valueerror(self, feature_container):
        with pytest.raises(ValueError):
            sc = SegmentContainer("fake_audio_path")
            sc.segments.append(Segment(0.5, 0.6))

            ffc_ext = FrameFeatureChunkExtractor("fake_feature")
            ffc_ext.execute(sc, feature_container)


class TestMelSpectrumExtractor:

    @pytest.fixture(scope="module")
    def config(self):
        return {
                "sample_rate": 22050,
                "fft_size": 256,
                "n_mels": 32,
                "min_freq": 0,
                "max_freq": 11025,
                "log_amp": False}
    
    def test_init(self, config):
        try:
            MelSpectrumExtractor(**config)
        except Exception as e:
            pytest.fail("Unexpected Error: {}".format(e))
    
    def test_execute(self, config):
        # trust librosa on mel calculation
        # just make sure we get the same values
        mel_ext = MelSpectrumExtractor(**config)
        data, sr = sf.read(os.path.join(*TEST_AUDIO_PATH_TUPLE))
        spec = np.abs(np.fft.rfft(data[:256])) ** 2
        mel = mel_ext.execute(spec)
        mel_librosa = melspectrogram(sr=22050, S=spec, n_fft=256,
                hop_length=128, n_mels=32, fmin=0, fmax=11025)
        assert np.allclose(mel, mel_librosa) # not exactly equals because
                                             # librosa uses float64


class TestMFCCExtractor:

    @pytest.fixture(scope="module")
    def mel_config(self):
        return {
                "sample_rate": 22050,
                "fft_size": 256,
                "n_mels": 128,
                "min_freq": 0,
                "max_freq": 11025,
                "log_amp": False}
    
    @pytest.fixture(scope="module")
    def mfcc_config(self, mel_config):
        mfcc_config = copy.deepcopy(mel_config)
        mfcc_config.pop("log_amp")
        mfcc_config["n_mfcc"] = 32
        return mfcc_config
    
    def test_init(self, mfcc_config):
        try:
            MFCCExtractor(**mfcc_config)
        except Exception as e:
            pytest.fail("Unexpected Error: {}".format(e))
    
    def test_execute(self, mel_config, mfcc_config):
        # trust librosa on MFCC calculation
        # just make sure we get the same values
        mel_ext = MelSpectrumExtractor(**mel_config)
        data, sr = sf.read(os.path.join(*TEST_AUDIO_PATH_TUPLE))
        spec = np.abs(np.fft.rfft(data[:256])) ** 2
        mfcc_ext = MFCCExtractor(**mfcc_config)
        mfcc = mfcc_ext.execute(spec)
        log_mel = logamplitude(mel_ext.execute(spec), top_db=None)
        mfcc_librosa = lr_mfcc(sr=22050, S=log_mel, n_fft=256,
                hop_length=128, n_mels=128, n_mfcc=32, fmin=0, fmax=11025,
                top_db=None)

        assert np.allclose(mfcc, mfcc_librosa) # not exactly equals because
                                             # librosa uses float64


class TestSpectralFlatnessExtractor:
    
    def test_init(self):
        try:
            SpectralFlatnessExtractor()
        except Exception as e:
            pytest.fail("Unexpected Error: {}".format(e))
    
    def test_execute(self):
        white_noise = np.ones((256,))
        tone = np.zeros((256,))
        tone[50] = 0.5

        sf_ext = SpectralFlatnessExtractor()
        assert (np.isclose(sf_ext.execute(white_noise), 1) and
            np.isclose(sf_ext.execute(tone), 0))


class TestFrameFeatureProcessor:

    @pytest.fixture(scope="module")
    def af_gen(self):
        win_size = 256
        hop_size = 128
        return AudioFrameGen(win_size, hop_size, Window.rect)
    
    @pytest.fixture(scope="module")
    def en_ext(self):
        return EnergyExtractor()
    
    @pytest.fixture(scope="module")
    def feature_container_root(self):
        return None

    def test_init(self, af_gen, en_ext, feature_container_root):
        try:
            FrameFeatureProcessor(af_gen, [en_ext], feature_container_root)
        except Exception as e:
            pytest.fail("Unexpected Error: {}".format(e))
    
    def test_execute(self, af_gen, en_ext):
        ff_pro = FrameFeatureProcessor(af_gen, [en_ext])
        fc, created = ff_pro.execute(TEST_AUDIO_PATH_TUPLE)
        assert (created and
                np.isclose(0.074310362339, fc.features["energy"]["data"][10]))

    def test_execute_typeerror(self, af_gen):
        with pytest.raises(TypeError):
            ac_ext = AudioChunkExtractor(22050)
            FrameFeatureProcessor(af_gen, [ac_ext])
    
    def test_execute_existing_fc(self, af_gen, en_ext):
        ff_pro = FrameFeatureProcessor(af_gen, [en_ext], DATA_PATH)
        fc, created = ff_pro.execute(TEST_AUDIO_PATH_TUPLE)
        assert (not created and
                np.isclose(0.074310362339, fc.features["energy"]["data"][10]))
    
    def test_execute_save_load_fc(self, tmpdir, af_gen, en_ext):
        fc_path = str(tmpdir)
        ff_pro = FrameFeatureProcessor(af_gen, [en_ext], fc_path)
        fc, created_1 = ff_pro.execute(TEST_AUDIO_PATH_TUPLE)
        fc, created_2 = ff_pro.execute(TEST_AUDIO_PATH_TUPLE)
        assert (created_1 and not created_2 and
                np.isclose(0.074310362339, fc.features["energy"]["data"][10]))

        
class TestSegmentFeatureProcessor:

    @pytest.fixture(scope="module")
    def ac_ext(self):
        sample_rate = 22050
        return AudioChunkExtractor(TEST_AUDIO_PATH_TUPLE[0], sample_rate)

    @pytest.fixture(scope="module")
    def segment_container(self):
        sc = SegmentContainer(TEST_AUDIO_PATH_TUPLE[1])
        sc.segments.append(Segment(0.5, 0.6))
        return sc

    def test_init(self, ac_ext):
        try:
            SegmentFeatureProcessor([ac_ext])
        except Exception as e:
            pytest.fail("Unexpected Error: {}".format(e))
    
    def test_execute(self, ac_ext, segment_container):
        data, sample_rate = sf.read(os.path.join(*TEST_AUDIO_PATH_TUPLE))
        sf_pro = SegmentFeatureProcessor([ac_ext])
        sf_pro.execute(segment_container)
        assert np.all(segment_container.segments[0].features["audio_chunk"] ==
                data[int(0.5 * 22050):int(0.6 * 22050)])

    def test_execute_typeerror(self):
        with pytest.raises(TypeError):
            en_ext = EnergyExtractor()
            SegmentFeatureProcessor([en_ext])
