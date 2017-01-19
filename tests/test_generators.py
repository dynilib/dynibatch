import os
import shutil
import pytest
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
from libdyni.features.activity_detection.simple import Simple
from libdyni.features.frame_feature_processor import FrameFeatureProcessor

from libdyni.utils import feature_container
from libdyni.utils import utils


DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

TEST_AUDIO_PATH_TUPLE_1 = (DATA_PATH, "ID0132.wav")
TEST_AUDIO_PATH_TUPLE_2 = (DATA_PATH, "ID0133.wav")
TEST_AUDIO_PATH_TUPLE_3 = (DATA_PATH, "ID1238.wav")
TEST_AUDIO_PATH_TUPLE_4 = (DATA_PATH, "ID1322.wav")
TEST_FILE2LABEL_PATH = os.path.join(DATA_PATH, "file2label.csv")

FEATURE_ROOT = os.path.join(DATA_PATH, "feature_root")

class TestAudioFrameGen:

    def test_init(self):
        try:
            sample_rate = 22050
            win_size = 256
            hop_size = 128
            AudioFrameGen(sample_rate, win_size, hop_size)
        except Exception as e:
            pytest.fail("Unexpected Error: {}".format(e))

    def test_execute(self):
        sample_rate = 22050
        win_size = 256
        hop_size = 128
        af_gen = AudioFrameGen(sample_rate, win_size, hop_size, win_type=Window.rect)
        af_gen_e = af_gen.execute(os.path.join(*TEST_AUDIO_PATH_TUPLE_1))
        next(af_gen_e) # 1st frame
        frame = next(af_gen_e) # 2nd frame
        data, _ = sf.read(os.path.join(*TEST_AUDIO_PATH_TUPLE_1))
        assert np.all(data[128:128+256] == frame)

    def test_wrong_sample_rate(self):
        sample_rate = 44100
        win_size = 256
        hop_size = 128
        af_gen = AudioFrameGen(sample_rate, win_size, hop_size, win_type=Window.rect)
        af_gen_e = af_gen.execute(os.path.join(*TEST_AUDIO_PATH_TUPLE_1))
        with pytest.raises(Exception) as e_info:
            next(af_gen_e) # 1st frame

class TestSegmentContainerGenerator:

    @pytest.fixture(scope="module")
    def ac_ext(self):
        sample_rate = 22050
        return AudioChunkExtractor(TEST_AUDIO_PATH_TUPLE_1[0], sample_rate)

    def test_init(self):
        try:
            parser = CSVLabelParser(TEST_FILE2LABEL_PATH)
            sf_pro = SegmentFeatureProcessor([])
            SegmentContainerGenerator("fake_audio_root",
                                      sf_pro,
                                      label_parser=parser)
        except Exception as e:
            pytest.fail("Unexpected Error: {}".format(e))

    def test_start(self):
        try:
            parser = CSVLabelParser(TEST_FILE2LABEL_PATH)
            sf_pro = SegmentFeatureProcessor([])
            sc_gen = SegmentContainerGenerator("fake_audio_root",
                                               sf_pro,
                                               label_parser=parser)
            sc_gen.start()
        except Exception as e:
            pytest.fail("Unexpected Error: {}".format(e))

    def test_execute(self, ac_ext):

        sample_rate = 22050
        seg_duration = 0.1
        seg_overlap = 0.5

        parser = CSVLabelParser(TEST_FILE2LABEL_PATH)
        sf_pro = SegmentFeatureProcessor([ac_ext])
        sc_gen = SegmentContainerGenerator(TEST_AUDIO_PATH_TUPLE_1[0],
                                           sf_pro,
                                           label_parser=parser,
                                           seg_duration=seg_duration,
                                           seg_overlap=seg_overlap)
        sc_gen.start()
        sc_list = [sc for sc in sc_gen.execute()]

        id0132_data, _ = sf.read(os.path.join(*TEST_AUDIO_PATH_TUPLE_1))
        id1238_data, _ = sf.read(os.path.join(*TEST_AUDIO_PATH_TUPLE_4))

        first_seg = sc_list[0].segments[0]
        first_seg_ref = id0132_data[:int(seg_duration*sample_rate)]
        last_seg = sc_list[-1].segments[-1]
        start_time = 0.0
        while start_time + seg_duration < len(id1238_data) / sample_rate:
            start_time += seg_duration * seg_overlap
        start_time -= seg_duration * seg_overlap
        last_seg_ref = \
            id1238_data[int(start_time*sample_rate):int((start_time+seg_duration)*sample_rate)]

        assert (len(sc_list) == 4 and
                np.all(first_seg_ref == first_seg.features["audio_chunk"]) and
                np.all(last_seg_ref == last_seg.features["audio_chunk"]))


class TestMiniBatch:

    @pytest.fixture(scope="module")
    def ac_ext(self):
        sample_rate = 22050
        return AudioChunkExtractor(DATA_PATH, sample_rate)

    def test_gen_minibatches_1d(self, ac_ext):
        sample_rate = 22050
        seg_duration = 0.1
        seg_overlap = 0.5
        seg_size = int(seg_duration * sample_rate)

        parser = CSVLabelParser(TEST_FILE2LABEL_PATH)
        classes = parser.get_labels()
        sf_pro = SegmentFeatureProcessor([ac_ext])
        sc_gen = SegmentContainerGenerator(DATA_PATH,
                                           sf_pro,
                                           label_parser=parser,
                                           seg_duration=seg_duration,
                                           seg_overlap=seg_overlap)

        id0132_data, _ = sf.read(os.path.join(*TEST_AUDIO_PATH_TUPLE_1))
        id0133_data, _ = sf.read(os.path.join(*TEST_AUDIO_PATH_TUPLE_2))
        id1238_data, _ = sf.read(os.path.join(*TEST_AUDIO_PATH_TUPLE_3))
        id1322_data, _ = sf.read(os.path.join(*TEST_AUDIO_PATH_TUPLE_4))

        n_epochs = 3
        batch_size = 10
        n_time_bins = int(seg_duration * sample_rate)

        chunk_size = sample_rate * seg_duration
        id0132_n_chunks = utils.get_n_overlapping_chunks(len(id0132_data),
                                                         chunk_size,
                                                         seg_overlap)
        id0133_n_chunks = utils.get_n_overlapping_chunks(len(id0133_data),
                                                         chunk_size,
                                                         seg_overlap)
        id1238_n_chunks = utils.get_n_overlapping_chunks(len(id1238_data),
                                                         chunk_size,
                                                         seg_overlap)
        id1322_n_chunks = utils.get_n_overlapping_chunks(len(id1322_data),
                                                         chunk_size,
                                                         seg_overlap)

        n_minibatches = (id0132_n_chunks + id0133_n_chunks +
                         id1238_n_chunks + id1322_n_chunks) // batch_size

        mb_gen = MiniBatchGen(sc_gen,
                              "audio_chunk",
                              batch_size,
                              1,
                              n_time_bins)

        for _ in range(n_epochs):
            mb_gen.reset()
            mb_gen_e = mb_gen.execute(with_targets=True,
                                      with_filenames=True)
            count = 0
            start_time = 0.0
            chunk_count = 0
            is_dataset1, is_dataset2, is_dataset3 = [True, True, True]
            for data, targets, filenames in mb_gen_e:
                for d, t, f in zip(data, targets, filenames):
                    start_ind = int(start_time * sample_rate)
                    if chunk_count < id0132_n_chunks:
                        assert f == "ID0132.wav"
                        assert t == classes.index("bird_c")
                        assert np.all(d == id0132_data[start_ind:start_ind+seg_size])
                    elif chunk_count < id0132_n_chunks + id0133_n_chunks:
                        if is_dataset1:
                            is_dataset1 = False
                            start_time = 0.0
                            start_ind = 0

                        assert f == "ID0133.wav"
                        assert t == classes.index("bird_c")
                        assert np.all(d == id0133_data[start_ind:start_ind+seg_size])
                    elif chunk_count < id0132_n_chunks + id0133_n_chunks + id1238_n_chunks:
                        if is_dataset2:
                            is_dataset2 = False
                            start_time = 0.0
                            start_ind = 0

                        assert f == "ID1238.wav"
                        assert t == classes.index("bird_d")
                        assert np.all(d == id1238_data[start_ind:start_ind+seg_size])
                    else:
                        if is_dataset3:
                            is_dataset3 = False
                            start_time = 0.0
                            start_ind = 0

                        assert f == "ID1322.wav"
                        assert t == classes.index("bird_d")
                        assert np.all(d == id1322_data[start_ind:start_ind+seg_size])

                    start_time += (1 - seg_overlap) * seg_duration
                    chunk_count += 1
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

        af_gen = AudioFrameGen(sample_rate=sample_rate, win_size=win_size, hop_size=hop_size)

        en_ext = EnergyExtractor()
        sf_ext = SpectralFlatnessExtractor()
        mel_ext = MelSpectrumExtractor(sample_rate=sample_rate,
                                       fft_size=win_size,
                                       n_mels=64,
                                       min_freq=0,
                                       max_freq=sample_rate/2)
        ff_pro = FrameFeatureProcessor(af_gen,
                                       [en_ext, sf_ext, mel_ext],
                                       DATA_PATH)

        pca = None
        scaler = None

        ffc_ext = FrameFeatureChunkExtractor("mel_spectrum", pca, scaler)
        act_det = Simple(energy_threshold=energy_threshold,
                         spectral_flatness_threshold=spectral_flatness_threshold)
        sf_pro = SegmentFeatureProcessor([act_det, ffc_ext],
                                         ff_pro=ff_pro,
                                         audio_root=DATA_PATH)

        parser = CSVLabelParser(TEST_FILE2LABEL_PATH)
        sc_gen = SegmentContainerGenerator(DATA_PATH,
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
            fc_path = os.path.join(DATA_PATH, sc.audio_path.replace(".wav", ".fc.jl"))
            fc = feature_container.FeatureContainer.load(fc_path)
            for s in sc.segments:
                if hasattr(s, 'activity') and s.activity:
                    start_ind = fc.time_to_frame_ind(s.start_time)
                    end_ind = start_ind + num_time_bins
                    data = fc.features["mel_spectrum"]["data"][start_ind:end_ind]
                    assert np.all(data == s.features["mel_spectrum"])
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
                assert np.all(data[0].T == active_segments[count].features["mel_spectrum"])
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

        af_gen = AudioFrameGen(sample_rate=sample_rate, win_size=win_size, hop_size=hop_size)

        en_ext = EnergyExtractor()
        sf_ext = SpectralFlatnessExtractor()
        mel_ext = MelSpectrumExtractor(sample_rate=sample_rate,
                                       fft_size=win_size,
                                       n_mels=64,
                                       min_freq=0,
                                       max_freq=sample_rate/2)
        ff_pro = FrameFeatureProcessor(af_gen,
                                       [en_ext, sf_ext, mel_ext],
                                       DATA_PATH)

        pca = None
        scaler = joblib.load(os.path.join(DATA_PATH, "transform/mel64_norm/scaler.jl"))

        ffc_ext = FrameFeatureChunkExtractor("mel_spectrum", pca, scaler)
        act_det = Simple(energy_threshold=energy_threshold,
                         spectral_flatness_threshold=spectral_flatness_threshold)
        sf_pro = SegmentFeatureProcessor([act_det, ffc_ext],
                                         ff_pro=ff_pro,
                                         audio_root=DATA_PATH)

        parser = CSVLabelParser(TEST_FILE2LABEL_PATH)
        sc_gen = SegmentContainerGenerator(DATA_PATH,
                                           sf_pro,
                                           label_parser=parser,
                                           seg_duration=seg_duration,
                                           seg_overlap=seg_overlap)

        sc_gen.reset()

        sc_gen_e = sc_gen.execute()

        active_segments = []

        # compare data in segment and corresponding data in feature container
        for sc in sc_gen_e:
            fc_path = os.path.join(DATA_PATH, sc.audio_path.replace(".wav", ".fc.jl"))
            fc = feature_container.FeatureContainer.load(fc_path)
            for s in sc.segments:
                if hasattr(s, 'activity') and s.activity:
                    start_ind = fc.time_to_frame_ind(s.start_time)
                    end_ind = start_ind + num_time_bins
                    data = scaler.transform(fc.features["mel_spectrum"]["data"][start_ind:end_ind])
                    assert np.all(data == s.features["mel_spectrum"])
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
                assert np.all(data[0].T == active_segments[count].features["mel_spectrum"])
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

        af_gen = AudioFrameGen(sample_rate=sample_rate, win_size=win_size, hop_size=hop_size)

        en_ext = EnergyExtractor()
        sf_ext = SpectralFlatnessExtractor()
        mel_ext = MelSpectrumExtractor(sample_rate=sample_rate,
                                       fft_size=win_size,
                                       n_mels=64,
                                       min_freq=0,
                                       max_freq=sample_rate/2)
        ff_pro = FrameFeatureProcessor(af_gen,
                                       [en_ext, sf_ext, mel_ext],
                                       DATA_PATH)

        pca = joblib.load(os.path.join(DATA_PATH, "transform/mel64_pca16_norm/pca.jl"))
        scaler = joblib.load(os.path.join(DATA_PATH, "transform/mel64_pca16_norm/scaler.jl"))

        ffc_ext = FrameFeatureChunkExtractor("mel_spectrum", pca, scaler)
        act_det = Simple(energy_threshold=energy_threshold,
                         spectral_flatness_threshold=spectral_flatness_threshold)
        sf_pro = SegmentFeatureProcessor([act_det, ffc_ext],
                                         ff_pro=ff_pro,
                                         audio_root=DATA_PATH)

        parser = CSVLabelParser(TEST_FILE2LABEL_PATH)
        sc_gen = SegmentContainerGenerator(DATA_PATH,
                                           sf_pro,
                                           label_parser=parser,
                                           seg_duration=seg_duration,
                                           seg_overlap=seg_overlap)

        sc_gen.reset()

        sc_gen_e = sc_gen.execute()

        active_segments = []

        # compare data in segment and corresponding data in feature container
        for sc in sc_gen_e:
            fc_path = os.path.join(DATA_PATH, sc.audio_path.replace(".wav", ".fc.jl"))
            fc = feature_container.FeatureContainer.load(fc_path)
            for s in sc.segments:
                if hasattr(s, 'activity') and s.activity:
                    start_ind = fc.time_to_frame_ind(s.start_time)
                    end_ind = start_ind + num_time_bins
                    data = scaler.transform(
                        pca.transform(fc.features["mel_spectrum"]["data"][start_ind:end_ind]))
                    assert np.all(data == s.features["mel_spectrum"])
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
                assert np.all(data[0].T == active_segments[count].features["mel_spectrum"])
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

    def test_consuming_minibatch(self):
        mb_gen_dict = MiniBatchGen.from_json_config_file("tests/config/config_test.json")
        try:
            if not os.path.exists(FEATURE_ROOT):
                os.makedirs(FEATURE_ROOT)
            for _, mb_gen in mb_gen_dict.items():
                mb_gen.start()
                mb_gen_e = mb_gen.execute()
                next(mb_gen_e)
        except Exception as e:
            pytest.fail("Unexpected Error: {}".format(e))
        finally:
            shutil.rmtree(FEATURE_ROOT)

    def test_data(self):
        """
        Compare data from "manual" constructor to config file-based constructor.
        """

        # construct minibatch generator "manually"

        sample_rate = 22050
        win_size = 256
        hop_size = 128
        energy_threshold = 0.2
        spectral_flatness_threshold = 0.3
        seg_duration = 0.2
        seg_overlap = 0.5

        batch_size = 50
        num_features = 64
        num_time_bins = 34

        af_gen = AudioFrameGen(sample_rate=sample_rate, win_size=win_size, hop_size=hop_size)

        en_ext = EnergyExtractor()
        sf_ext = SpectralFlatnessExtractor()
        mel_ext = MelSpectrumExtractor(sample_rate=sample_rate,
                                       fft_size=win_size,
                                       n_mels=64,
                                       min_freq=0,
                                       max_freq=sample_rate/2)
        ff_pro = FrameFeatureProcessor(af_gen,
                                       [en_ext, sf_ext, mel_ext],
                                       DATA_PATH)

        ffc_ext = FrameFeatureChunkExtractor("mel_spectrum")
        act_det = Simple(energy_threshold=energy_threshold,
                         spectral_flatness_threshold=spectral_flatness_threshold)
        sf_pro = SegmentFeatureProcessor([act_det, ffc_ext],
                                         ff_pro=ff_pro,
                                         audio_root=DATA_PATH)

        parser = CSVLabelParser(TEST_FILE2LABEL_PATH)
        sc_gen = SegmentContainerGenerator(DATA_PATH,
                                           sf_pro,
                                           label_parser=parser,
                                           seg_duration=seg_duration,
                                           seg_overlap=seg_overlap)

        mb_gen_1 = MiniBatchGen(sc_gen,
                                "mel_spectrum",
                                batch_size,
                                num_features,
                                num_time_bins)


        # construct minibatch generator from config file
        mb_gen_dict = MiniBatchGen.from_json_config_file("tests/config/config_test.json")
        mb_gen_2 = mb_gen_dict["default"]

        # execute and compare
        try:
            mb_gen_1.start()
            mb_gen_1_e = mb_gen_1.execute(active_segments_only=True,
                                          with_targets=True,
                                          with_filenames=False)

            mb_gen_2.start()
            mb_gen_2_e = mb_gen_2.execute(active_segments_only=True,
                                          with_targets=True,
                                          with_filenames=False)

            if not os.path.exists(FEATURE_ROOT):
                os.makedirs(FEATURE_ROOT)

            for mb1, mb2 in zip(mb_gen_1_e, mb_gen_2_e):
                assert np.all(mb1[0] == mb2[0])
                assert np.all(mb1[1] == mb2[1])

        except Exception as e:
            pytest.fail("Unexpected Error: {}".format(e))

        finally:
            shutil.rmtree(FEATURE_ROOT)
