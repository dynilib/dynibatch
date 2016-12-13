import pytest
import os
import joblib

import numpy as np
import soundfile as sf

from libdyni.generators.audio_frame_gen import AudioFrameGen
from libdyni.generators.audio_frame_gen import Window
from libdyni.generators.segment_container_gen import SegmentContainerGenerator
from libdyni.parsers.label_parsers import CSVLabelParser
from libdyni.features.segment_feature_processor import SegmentFeatureProcessor
from libdyni.features.extractors.audio_chunk import AudioChunkExtractor
from libdyni.generators.minibatch_gen import MiniBatchGen

from libdyni.features.extractors.energy import EnergyExtractor
from libdyni.features.extractors.spectral_flatness import SpectralFlatnessExtractor
from libdyni.features.extractors.mel_spectrum import MelSpectrumExtractor
from libdyni.features.extractors.frame_feature_chunk import FrameFeatureChunkExtractor
from libdyni.features.extractors.activity_detection import ActivityDetection
from libdyni.features.frame_feature_processor import FrameFeatureProcessor

from libdyni.utils import feature_container
from libdyni.utils import utils


DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
REDUCED_DATA_PATH = os.path.join(DATA_PATH, "reduced_set")
FULL_DATA_PATH = os.path.join(DATA_PATH, "full_set")

TEST_AUDIO_PATH_TUPLE_1 = (REDUCED_DATA_PATH, "ID0132.wav")
TEST_AUDIO_PATH_TUPLE_2 = (REDUCED_DATA_PATH, "ID1238.wav")
TEST_CSVLABEL_PATH = os.path.join(REDUCED_DATA_PATH, "labels.csv")

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


class TestMiniBatch:

    @pytest.fixture(scope="module")
    def ac_ext(self):
        sample_rate = 22050
        return AudioChunkExtractor(REDUCED_DATA_PATH, sample_rate)

    def test_gen_minibatches_1d(self, ac_ext):
        sample_rate = 22050
        seg_duration = 0.1
        seg_overlap = 0.5
        seg_size = int(seg_duration * sample_rate)
        hop_size = int(seg_duration * (1 - seg_overlap) * sample_rate)

        parser = CSVLabelParser(TEST_CSVLABEL_PATH)
        classes = parser.get_labels()
        sf_pro = SegmentFeatureProcessor([ac_ext])
        sc_gen = SegmentContainerGenerator(
                REDUCED_DATA_PATH,
                sf_pro,
                label_parser=parser,
                seg_duration=seg_duration,
                seg_overlap=seg_overlap)

        id0132_data, sr = sf.read(os.path.join(*TEST_AUDIO_PATH_TUPLE_1))
        id1238_data, sr = sf.read(os.path.join(*TEST_AUDIO_PATH_TUPLE_2))

        n_epochs = 3
        batch_size = 10
        n_time_bins = int(seg_duration * sample_rate)

        id0132_n_chunks = utils.get_n_overlapping_chunks(
                len(id0132_data),
                sample_rate * seg_duration,
                seg_overlap)
        id1238_n_chunks = utils.get_n_overlapping_chunks(
                len(id1238_data),
                sample_rate * seg_duration,
                seg_overlap)

        id0132_n_minibatches = int(id0132_n_chunks / batch_size)
        n_minibatches = int((id0132_n_chunks + id1238_n_chunks) / batch_size)

        mb_gen = MiniBatchGen(sc_gen,
                              "audio_chunk",
                              batch_size,
                              1,
                              n_time_bins)

        for i in range(n_epochs):
            mb_gen.reset()
            mb_gen_e = mb_gen.execute(with_targets=True,
                                      with_filenames=True)
            count = 0
            start_time = 0.0
            for data, target, filenames in mb_gen_e:
                if count < id0132_n_minibatches:
                    for d, t, f in zip(data, target, filenames):
                        start_ind = int(start_time * sample_rate)
                        assert np.all(d==id0132_data[start_ind:start_ind+seg_size])
                        assert t == classes["bird_c"]
                        assert f == "ID0132.wav"
                        start_time += (1 - seg_overlap) * seg_duration
                elif count == id0132_n_minibatches:
                    start_time_reset = False
                    for i, d in enumerate(zip(data, target, filenames)):
                        if i < id0132_n_chunks % batch_size:
                            start_ind = int(start_time * sample_rate)
                            assert np.all(d[0]==id0132_data[start_ind:start_ind+seg_size])
                            assert d[1] == classes["bird_c"]
                            assert d[2] == "ID0132.wav"
                        else:
                            if not start_time_reset:
                                start_time = 0.0
                                start_time_reset = True
                            start_ind = int(start_time * sample_rate)
                            assert np.all(d[0]==id1238_data[start_ind:start_ind+seg_size])
                            assert d[1] == classes["bird_d"]
                            assert d[2] == "ID1238.wav"
                        start_time += (1 - seg_overlap) * seg_duration
                else:
                    for d, t, f in zip(data, target, filenames):
                        start_ind = int(start_time * sample_rate)
                        assert np.all(d==id1238_data[start_ind:start_ind+seg_size])
                        assert t == classes["bird_d"]
                        assert f == "ID1238.wav"
                        start_time += (1 - seg_overlap) * seg_duration
                count += 1

            assert count == n_minibatches


    def test_gen_minibatches_2d(self):

        sample_rate = 22050
        win_size = 256
        hop_size = 128
        energy_threshold = 0.2
        spectral_flatness_threshold = 0.3
        seg_duration = 0.1
        seg_overlap = 0.5

        batch_size = 10
        num_features = 64
        num_time_bins = 17

        af_gen = AudioFrameGen(win_size=win_size, hop_size=hop_size)

        en_ext = EnergyExtractor()
        sf_ext = SpectralFlatnessExtractor()
        mel_ext = MelSpectrumExtractor(
                sample_rate=sample_rate,
                fft_size=win_size,
                n_mels=64,
                min_freq=0,
                max_freq=sample_rate/2)
        ff_pro = FrameFeatureProcessor(
                af_gen,
                [en_ext, sf_ext, mel_ext],
                FULL_DATA_PATH)

        pca = None
        scaler = None

        ffc_ext = FrameFeatureChunkExtractor("mel_spectrum", pca, scaler)
        act_det = ActivityDetection(
                energy_threshold=energy_threshold,
                spectral_flatness_threshold=spectral_flatness_threshold)
        sf_pro = SegmentFeatureProcessor(
                [act_det, ffc_ext],
                ff_pro=ff_pro,
                audio_root=FULL_DATA_PATH)

        parser = CSVLabelParser(TEST_CSVLABEL_PATH)
        sc_gen = SegmentContainerGenerator(
               FULL_DATA_PATH,
               sf_pro,
               label_parser=parser,
               seg_duration=seg_duration,
               seg_overlap=seg_overlap)

        sc_gen.reset()

        sc_gen_e = sc_gen.execute()

        active_segments = []
        labels = []

        # compare data in segment and corresponding data in feature container
        for sc in sc_gen_e:
            fc_path = os.path.join(FULL_DATA_PATH, sc.audio_path.replace(".wav",
                ".fc.jl"))
            fc = feature_container.FeatureContainer.load(fc_path)
            for s in sc.segments:
                if hasattr(s, 'activity') and s.activity:
                    start_ind = fc.time_to_frame_ind(s.start_time)
                    end_ind = start_ind + num_time_bins
                    data = fc.features["mel_spectrum"]["data"][start_ind:end_ind]
                    assert np.all(data==s.features["mel_spectrum"])
                    active_segments.append(s)
                    labels.append(s.label)

        # compare data in segment and corresponding data in minibatches
        #classes = parser.get_labels()

        mb_gen = MiniBatchGen(sc_gen,
                              "mel_spectrum",
                              batch_size,
                              num_features,
                              num_time_bins)

        mb_gen.start()

        mb_gen_e = mb_gen.execute(active_segments_only=True,
                                  with_targets=True,
                                  with_filenames=False)

        count = 0
        for mb in mb_gen_e:
            for data, target in zip(*mb):
                assert np.all(data[0].T==active_segments[count].features["mel_spectrum"])
                assert target == labels[count]
                count += 1


    def test_gen_minibatches_2d_w_scaler(self):

        sample_rate = 22050
        win_size = 256
        hop_size = 128
        energy_threshold = 0.2
        spectral_flatness_threshold = 0.3
        seg_duration = 0.1
        seg_overlap = 0.5

        batch_size = 10
        num_features = 64
        num_time_bins = 17

        af_gen = AudioFrameGen(win_size=win_size, hop_size=hop_size)

        en_ext = EnergyExtractor()
        sf_ext = SpectralFlatnessExtractor()
        mel_ext = MelSpectrumExtractor(
                sample_rate=sample_rate,
                fft_size=win_size,
                n_mels=64,
                min_freq=0,
                max_freq=sample_rate/2)
        ff_pro = FrameFeatureProcessor(
                af_gen,
                [en_ext, sf_ext, mel_ext],
                FULL_DATA_PATH)

        pca = None
        scaler = joblib.load(os.path.join(DATA_PATH, "transform/mel64_norm/scaler.jl"))

        ffc_ext = FrameFeatureChunkExtractor("mel_spectrum", pca, scaler)
        act_det = ActivityDetection(
                energy_threshold=energy_threshold,
                spectral_flatness_threshold=spectral_flatness_threshold)
        sf_pro = SegmentFeatureProcessor(
                [act_det, ffc_ext],
                ff_pro=ff_pro,
                audio_root=FULL_DATA_PATH)

        parser = CSVLabelParser(TEST_CSVLABEL_PATH)
        sc_gen = SegmentContainerGenerator(
               FULL_DATA_PATH,
               sf_pro,
               label_parser=parser,
               seg_duration=seg_duration,
               seg_overlap=seg_overlap)

        sc_gen.reset()

        sc_gen_e = sc_gen.execute()

        active_segments = []

        # compare data in segment and corresponding data in feature container
        for sc in sc_gen_e:
            fc_path = os.path.join(FULL_DATA_PATH, sc.audio_path.replace(".wav",
                ".fc.jl"))
            fc = feature_container.FeatureContainer.load(fc_path)
            for s in sc.segments:
                if hasattr(s, 'activity') and s.activity:
                    start_ind = fc.time_to_frame_ind(s.start_time)
                    end_ind = start_ind + num_time_bins
                    data = scaler.transform(fc.features["mel_spectrum"]["data"][start_ind:end_ind])
                    assert np.all(data==s.features["mel_spectrum"])
                    active_segments.append(s)

        # compare data in segment and corresponding data in minibatches

        mb_gen = MiniBatchGen(sc_gen,
                              "mel_spectrum",
                              batch_size,
                              num_features,
                              num_time_bins)

        mb_gen.start()

        mb_gen_e = mb_gen.execute(active_segments_only=True,
                                  with_targets=False,
                                  with_filenames=False)

        count = 0
        for mb, in mb_gen_e:
            for data in mb:
                assert np.all(data[0].T==active_segments[count].features["mel_spectrum"])
                count += 1


    def test_gen_minibatches_2d_w_pca_scaler(self):

        sample_rate = 22050
        win_size = 256
        hop_size = 128
        energy_threshold = 0.2
        spectral_flatness_threshold = 0.3
        seg_duration = 0.1
        seg_overlap = 0.5

        batch_size = 10
        num_features = 16
        num_time_bins = 17

        af_gen = AudioFrameGen(win_size=win_size, hop_size=hop_size)

        en_ext = EnergyExtractor()
        sf_ext = SpectralFlatnessExtractor()
        mel_ext = MelSpectrumExtractor(
                sample_rate=sample_rate,
                fft_size=win_size,
                n_mels=64,
                min_freq=0,
                max_freq=sample_rate/2)
        ff_pro = FrameFeatureProcessor(
                af_gen,
                [en_ext, sf_ext, mel_ext],
                FULL_DATA_PATH)

        pca = joblib.load(os.path.join(DATA_PATH, "transform/mel64_pca16_norm/pca.jl"))
        scaler = joblib.load(os.path.join(DATA_PATH, "transform/mel64_pca16_norm/scaler.jl"))

        ffc_ext = FrameFeatureChunkExtractor("mel_spectrum", pca, scaler)
        act_det = ActivityDetection(
                energy_threshold=energy_threshold,
                spectral_flatness_threshold=spectral_flatness_threshold)
        sf_pro = SegmentFeatureProcessor(
                [act_det, ffc_ext],
                ff_pro=ff_pro,
                audio_root=FULL_DATA_PATH)

        parser = CSVLabelParser(TEST_CSVLABEL_PATH)
        sc_gen = SegmentContainerGenerator(
               FULL_DATA_PATH,
               sf_pro,
               label_parser=parser,
               seg_duration=seg_duration,
               seg_overlap=seg_overlap)

        sc_gen.reset()

        sc_gen_e = sc_gen.execute()

        active_segments = []

        # compare data in segment and corresponding data in feature container
        for sc in sc_gen_e:
            fc_path = os.path.join(FULL_DATA_PATH, sc.audio_path.replace(".wav",
                ".fc.jl"))
            fc = feature_container.FeatureContainer.load(fc_path)
            for s in sc.segments:
                if hasattr(s, 'activity') and s.activity:
                    start_ind = fc.time_to_frame_ind(s.start_time)
                    end_ind = start_ind + num_time_bins
                    data = scaler.transform(pca.transform(fc.features["mel_spectrum"]["data"][start_ind:end_ind]))
                    assert np.all(data==s.features["mel_spectrum"])
                    active_segments.append(s)

        # compare data in segment and corresponding data in minibatches

        mb_gen = MiniBatchGen(sc_gen,
                              "mel_spectrum",
                              batch_size,
                              num_features,
                              num_time_bins)

        mb_gen.start()

        mb_gen_e = mb_gen.execute(active_segments_only=True,
                                  with_targets=False,
                                  with_filenames=False)

        count = 0
        for mb, in mb_gen_e:
            for data in mb:
                assert np.all(data[0].T==active_segments[count].features["mel_spectrum"])
                count += 1

class TestMiniBatchGenFromConfig:
    """
        Test Minibatch instance creation from json config file
    """

    def test_init(self):
        try:
            MiniBatchGen.from_json_config_file("tests/config/config_test.json")
        except Exception as e:
            pytest.fail("Unexpected Error: {}".format(e))

    def test_get_minibatch(self):
        mb_gen = MiniBatchGen.from_json_config_file("tests/config/config_test.json")
        mb_gen.start()
        try:
            mb_gen.execute()
        except Exception as e:
            pytest.fail("Unexpected Error: {}".format(e))
