import os
import pytest
import random
import joblib

import numpy as np
import soundfile as sf

from libdyni.utils.exceptions import ParameterError
from libdyni.utils import segment
from libdyni.utils import segment_container
from libdyni.utils import feature_container
from libdyni.utils import datasplit_utils
from libdyni.utils import utils
from libdyni.utils.minibatch_gen import MiniBatchGen
from libdyni.generators.audio_frame_gen import AudioFrameGen
from libdyni.generators.segment_container_gen import SegmentContainerGenerator
from libdyni.utils.label_parsers import CSVLabelParser
from libdyni.features.extractors.energy import EnergyExtractor
from libdyni.features.extractors.spectral_flatness import SpectralFlatnessExtractor
from libdyni.features.extractors.mel_spectrum import MelSpectrumExtractor
from libdyni.features.extractors.frame_feature_chunk import FrameFeatureChunkExtractor
from libdyni.features.extractors.audio_chunk import AudioChunkExtractor
from libdyni.features.extractors.activity_detection import ActivityDetection
from libdyni.features.frame_feature_processor import FrameFeatureProcessor
from libdyni.features.segment_feature_processor import SegmentFeatureProcessor

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
REDUCED_DATA_PATH = os.path.join(DATA_PATH, "reduced_set")
FULL_DATA_PATH = os.path.join(DATA_PATH, "full_set")

TEST_AUDIO_PATH_TUPLE_1 = (REDUCED_DATA_PATH, "ID0132.wav")
TEST_AUDIO_PATH_TUPLE_2 = (REDUCED_DATA_PATH, "ID1238.wav")
TEST_SEG_PATH_TUPLE_1 = (REDUCED_DATA_PATH, "ID0132.seg")
TEST_CSVLABEL_PATH = os.path.join(REDUCED_DATA_PATH, "labels.csv")
TEST_DURATION = 15.45
TEST_N_SEGMENTS = 4
TEST_FIRST_SEGMENT_DURATION = 0.79


class TestSegment:

    def test_init(self):
        try:
            start_time = 1
            end_time = 10
            segment.Segment(start_time, end_time)
        except ParameterError:
            pytest.fail("Unexpected ParameterError")

    def test_negative_start_time(self):
        with pytest.raises(ParameterError):
            start_time = -1
            end_time = 10
            segment.Segment(start_time, end_time)

    def test_time_order(self):
        with pytest.raises(ParameterError):
            start_time = 3
            end_time = 1
            segment.Segment(start_time, end_time)

    def test_set_segment_labels(self):
        segment_from_list = []
        segment_from_list.append(segment.Segment(0, 1, "a"))
        segment_from_list.append(segment.Segment(1.2, 2, "b"))
        segment_to_list = []
        segment_to_list.append(segment.Segment(0.2, 0.3))
        segment_to_list.append(segment.Segment(0.6, 1.1))
        segment_to_list.append(segment.Segment(1.7, 1.8))
        segment_to_list.append(segment.Segment(2.5, 2.7))
        segment.set_segment_labels(
                segment_from_list,
                segment_to_list,
                overlap_ratio=0.5)
        assert (segment_to_list[0].label == "a" and
            segment_to_list[1].label == "a" and
            segment_to_list[2].label == "b" and
            segment_to_list[3].label == segment.CommonLabels.unknown)

    def test_set_segment_labels_overlap(self):
        segment_from_list = []
        segment_from_list.append(segment.Segment(0, 1, "a"))
        segment_to_list = []
        segment_to_list.append(segment.Segment(0.4, 1.5))
        segment_to_list.append(segment.Segment(0.6, 1.5))
        segment.set_segment_labels(
                segment_from_list,
                segment_to_list,
                overlap_ratio=0.5)
        assert (segment_to_list[0].label == "a" and
            segment_to_list[1].label == segment.CommonLabels.unknown)


class TestSegmentContainer:

    def test_no_segments_n_segments(self):
        sc = segment_container.SegmentContainer("fake_audio_path")
        assert sc.n_segments == 0

    def test_no_segments_n_active_segments(self):
        sc = segment_container.SegmentContainer("fake_audio_path")
        assert sc.n_active_segments == 0

    def test_no_segments_n_segments_w_label(self):
        sc = segment_container.SegmentContainer("fake_audio_path")
        assert sc.n_segments_with_label("fake_label") == 0

    def test_no_segments_n_active_segments_w_label(self):
        sc = segment_container.SegmentContainer("fake_audio_path")
        assert sc.n_active_segments_with_label("fake_label") == 0

    def test_no_segments_features(self):
        sc = segment_container.SegmentContainer("fake_audio_path")
        assert not any(sc.has_features(["fake_feature"]))

    def test_n_segments(self):
        sc = segment_container.SegmentContainer("fake_audio_path")
        n_segments = 5
        for i in range(n_segments):
            sc.segments.append(segment.Segment(i * 1, (i + 1) * 1))
        assert sc.n_segments == n_segments

    def test_n_active_segments(self):
        sc = segment_container.SegmentContainer("fake_audio_path")
        n_segments = 5
        active_segment_ind = [2, 3]
        for i in range(n_segments):
            sc.segments.append(segment.Segment(i * 1, (i + 1) * 1))
            if i in active_segment_ind:
                sc.segments[-1].activity = True
        assert sc.n_active_segments == len(active_segment_ind)

    def test_no_active_segments(self):
        sc = segment_container.SegmentContainer("fake_audio_path")
        n_segments = 5
        for i in range(n_segments):
            sc.segments.append(segment.Segment(i * 1, (i + 1) * 1))
        assert sc.n_active_segments == 0

    def test_n_segments_w_label(self):
        sc = segment_container.SegmentContainer("fake_audio_path")
        n_segments = 5
        for i in range(n_segments):
            sc.segments.append(segment.Segment(i * 1, (i + 1) * 1))
        assert sc.n_segments_with_label(segment.CommonLabels.unknown) == n_segments

    def test_n_active_segments_w_labels(self):
        sc = segment_container.SegmentContainer("fake_audio_path")
        n_segments = 5
        active_segment_ind = [2, 3]
        for i in range(n_segments):
            sc.segments.append(segment.Segment(i * 1, (i + 1) * 1))
            if i in active_segment_ind:
                sc.segments[-1].activity = True
        assert sc.n_active_segments_with_label(segment.CommonLabels.unknown) == len(active_segment_ind)

    def test_create_segment_containers_from_audio_file_tuple(self):
        with pytest.raises(TypeError):
            segment_container.create_segment_containers_from_audio_file(
                os.path.join(*TEST_AUDIO_PATH_TUPLE_1))

    def test_create_segment_containers_from_audio_file_n_segment(self):
        sc = segment_container.create_segment_containers_from_audio_file(
            TEST_AUDIO_PATH_TUPLE_1)
        assert sc.n_segments == 1

    def test_create_segment_containers_from_audio_file_segment_duration(self):
        sc = segment_container.create_segment_containers_from_audio_file(
            TEST_AUDIO_PATH_TUPLE_1)
        assert np.abs(sc.segments[0].duration - TEST_DURATION) < 1e-03

    def test_create_segment_containers_from_seg_file_tuple(self):
        with pytest.raises(TypeError):
            segment_container.create_segment_containers_from_seg_file(
                os.path.join(*TEST_SEG_PATH_TUPLE_1))

    def test_create_segment_containers_from_seg_file_n_segment(self):
        sc = segment_container.create_segment_containers_from_seg_file(
            TEST_SEG_PATH_TUPLE_1)
        assert sc.n_segments == TEST_N_SEGMENTS

    def test_create_segment_containers_from_seg_file_segment_duration(self):
        sc = segment_container.create_segment_containers_from_seg_file(
            TEST_SEG_PATH_TUPLE_1)
        assert np.abs(sc.segments[0].duration - TEST_FIRST_SEGMENT_DURATION) < 1e-03

    def test_create_fixed_duration_segments_duration(self):
        file_duration = 12.5
        seg_duration = 0.4
        seg_overlap = 0.3
        segments = segment_container.create_fixed_duration_segments(file_duration, seg_duration,
                seg_overlap)
        assert np.all(np.isclose(np.asarray([s.duration for s in segments]),
                                 seg_duration))

    def test_create_fixed_duration_segments_n_segments(self):
        file_duration = 12.5
        seg_duration = 0.4
        seg_overlap = 0.3
        segments = segment_container.create_fixed_duration_segments(file_duration, seg_duration,
            seg_overlap)
        assert len(segments) == utils.get_n_overlapping_chunks(
            file_duration,
            seg_duration,
            seg_overlap)

    def test_parse_segment_file_line(self):
        line = "0.12; 0.15 ;  bird_a  "
        start_time, end_time, label = (
            segment_container._parse_segment_file_line(line, ";"))
        assert (np.isclose(start_time, 0.12) and np.isclose(end_time, 0.15) and
                label == "bird_a")


class TestFeatureContainer:

    def test_init(self):
        try:
            feature_container.FeatureContainer(
                    "fake_audio_path",
                    22050,
                    256,
                    128)
        except:
            pytest.fail("Unexpected Error")

    def test_features_ok(self):
        features = ["feat1", "feat2"]
        configs = ["config1", "config2"]
        fc =  feature_container.FeatureContainer(
                    "fake_audio_path",
                    22050,
                    256,
                    128)
        fc.features["feat1"]["data"] = np.random.sample(10)
        fc.features["feat1"]["config"] = "config1"
        fc.features["feat2"]["data"] = np.random.sample(10)
        fc.features["feat2"]["config"] = "config2"
        assert all(fc.has_features(list(zip(features, configs))))

    def test_features_wrong_features(self):
        features = ["feat1", "feat3"]
        configs = ["config1", "config2"]
        fc =  feature_container.FeatureContainer(
                    "fake_audio_path",
                    22050,
                    256,
                    128)
        fc.features["feat1"]["data"] = np.random.sample(10)
        fc.features["feat1"]["config"] = "config1"
        fc.features["feat2"]["data"] = np.random.sample(10)
        fc.features["feat2"]["config"] = "config2"
        assert not all(fc.has_features(list(zip(features, configs))))

    def test_features_wrong_configs(self):
        features = ["feat1", "feat2"]
        configs = ["config1", "config3"]
        fc =  feature_container.FeatureContainer(
                    "fake_audio_path",
                    22050,
                    256,
                    128)
        fc.features["feat1"]["data"] = np.random.sample(10)
        fc.features["feat1"]["config"] = "config1"
        fc.features["feat2"]["data"] = np.random.sample(10)
        fc.features["feat2"]["config"] = "config2"
        assert not all(fc.has_features(list(zip(features, configs))))

    def test_features_empty_features(self):
        features = ["feat1", "feat2"]
        configs = ["config1", "config2"]
        fc =  feature_container.FeatureContainer(
                    "fake_audio_path",
                    22050,
                    256,
                    128)
        fc.features["feat1"]["data"] = np.random.sample(10)
        fc.features["feat1"]["config"] = "config1"
        fc.features["feat2"]["config"] = "config2"
        assert not all(fc.has_features(list(zip(features, configs))))

    def test_time_to_frame_ind(self):
        sample_rate = 22050
        win_size = 256
        hop_size = 128
        fc =  feature_container.FeatureContainer(
                    "fake_audio_path",
                    sample_rate,
                    win_size,
                    hop_size)
        assert fc.time_to_frame_ind(0.015) == 2


class TestUtils:

    def test_get_n_overlapping_chunks(self):
        file_duration = 12.5
        seg_duration = 0.4
        seg_overlap = 0.3

        start = 0
        hop = seg_duration * (1 - seg_overlap)
        n_chunks = 0
        while start + seg_duration < file_duration:
            start += hop
            n_chunks += 1

        assert utils.get_n_overlapping_chunks(
                file_duration,
                seg_duration,
                seg_overlap) == n_chunks


class TestDatasplit:

    @pytest.fixture(scope="module")
    def n_files(self):
        return 1000

    @pytest.fixture(scope="module")
    def n_classes(self):
        return 10

    @pytest.fixture(scope="module")
    def n_files_per_class(self, n_files, n_classes):
        return int(n_files / n_classes)

    @pytest.fixture(scope="module")
    def file_list(self, n_files):
        return ["f{}".format(i) for i in range(n_files)]

    @pytest.fixture(scope="module")
    def label_list(self, n_files, n_files_per_class):
        return [int(i / n_files_per_class) for i in range(n_files)]

    @pytest.fixture(scope="module")
    def sc_list(self, n_files, file_list, label_list):
        sc_list = []
        for i in range(n_files):
            sc = segment_container.SegmentContainer(file_list[i])
            sc.segments.append(segment.Segment(0, 1, label_list[i]))
            sc_list.append(sc)
        return sc_list

    def test_create_datasplit_init(self, file_list):
        try:
            file_set = set(file_list)
            train_set = set(random.sample(file_set, 700))
            validation_set = set(random.sample(file_set-train_set, 100))
            test_set = set(random.sample(file_set-train_set-validation_set, 200))
            datasplit_utils.create_datasplit(train_set, validation_set, test_set,
                    name="fake_datasplit")
        except Exception as e:
            pytest.fail("Unexpected Error: {}".format(e))

    def test_create_datasplit_count(self, file_list):
        file_set = set(file_list)
        train_set = set(random.sample(file_set, 700))
        validation_set = set(random.sample(file_set-train_set, 100))
        test_set = set(random.sample(file_set-train_set-validation_set, 200))
        ds = datasplit_utils.create_datasplit(train_set, validation_set, test_set,
                name="fake_datasplit")
        assert (len(ds["sets"]["train"]) == 700 and
                len(ds["sets"]["validation"]) == 100 and
                len(ds["sets"]["test"]) == 200)

    def test_create_random_datasplit_init(self, sc_list):
        try:
            datasplit_utils.create_random_datasplit(
                    sc_list,
                    train_ratio=0.7,
                    validation_ratio=0.1,
                    test_ratio=0.2)
        except Exception as e:
            pytest.fail("Unexpected Error: {}".format(e))

    def test_create_random_datasplit_set_dont_sumup_to_one(self, sc_list):
        with pytest.raises(ParameterError):
            datasplit_utils.create_random_datasplit(
                    sc_list,
                    train_ratio=0.8,
                    validation_ratio=0.1,
                    test_ratio=0.2)

    def test_create_random_datasplit_count(self, sc_list):
        ds = datasplit_utils.create_random_datasplit(
                    sc_list,
                    train_ratio=0.7,
                    validation_ratio=0.1,
                    test_ratio=0.2)
        assert (len(ds["sets"]["train"]) == 700 and
                len(ds["sets"]["validation"]) == 100 and
                len(ds["sets"]["test"]) == 200)

    def test_datasplit_stats(self):
        # TODO (jul)
        pass


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

        mb_gen = MiniBatchGen("audio_chunk",
                batch_size,
                1,
                n_time_bins)

        for i in range(n_epochs):
            sc_gen.reset()
            mb_gen_e = mb_gen.execute(sc_gen,
                    with_targets=True,
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

        mb_gen = MiniBatchGen("mel_spectrum",
                batch_size,
                num_features,
                num_time_bins)
        
        sc_gen.reset()

        minibatches = []

        mb_gen_e = mb_gen.execute(sc_gen,
                active_segments_only=True,
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

        mb_gen = MiniBatchGen("mel_spectrum",
                batch_size,
                num_features,
                num_time_bins)
        
        sc_gen.reset()

        minibatches = []

        mb_gen_e = mb_gen.execute(sc_gen,
                active_segments_only=True,
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

        mb_gen = MiniBatchGen("mel_spectrum",
                batch_size,
                num_features,
                num_time_bins)
        
        sc_gen.reset()

        minibatches = []

        mb_gen_e = mb_gen.execute(sc_gen,
                active_segments_only=True,
                with_targets=False,
                with_filenames=False)

        count = 0
        for mb, in mb_gen_e:
            for data in mb:
                assert np.all(data[0].T==active_segments[count].features["mel_spectrum"])
                count += 1



