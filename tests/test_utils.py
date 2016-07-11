import os
import pytest
import random

import numpy as np
import soundfile as sf
from librosa.feature.spectral import melspectrogram

from libdyni.utils.exceptions import ParameterError
from libdyni.utils import segment
from libdyni.utils import segment_container
from libdyni.utils import feature_container
from libdyni.utils import datasplit_utils
from libdyni.utils import utils
from libdyni.utils.minibatch_gen import MiniBatchGen
from libdyni.generators.segment_container_gen import SegmentContainerGenerator
from libdyni.parsers.label_parsers import CSVLabelParser
from libdyni.generators.audio_frame_gen import AudioFrameGen
from libdyni.generators.audio_frame_gen import Window
from libdyni.features.extractors.audio_chunk import AudioChunkExtractor
from libdyni.features.extractors.mel_spectrum import MelSpectrumExtractor
from libdyni.features.frame_feature_processor import FrameFeatureProcessor
from libdyni.features.segment_feature_processor import SegmentFeatureProcessor
from libdyni.features.extractors.frame_feature_chunk import FrameFeatureChunkExtractor

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

TEST_AUDIO_PATH_TUPLE_1 = (DATA_PATH, "ID0132.wav")
TEST_AUDIO_PATH_TUPLE_2 = (DATA_PATH, "ID1238.wav")
TEST_SEG_PATH_TUPLE_1 = (DATA_PATH, "ID0132.seg")
TEST_CSVLABEL_PATH = os.path.join(DATA_PATH, "labels.csv")
TEST_DURATION = 15.45
TEST_N_SEGMENTS = 4
TEST_SEGMENT_DURATIONS = [0.79, 0.655, 0.686, 0.655]


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
            segment_to_list[3].label == segment.common_labels.unknown)

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
            segment_to_list[1].label == segment.common_labels.unknown)


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
        assert sc.n_segments_with_label(segment.common_labels.unknown) == n_segments

    def test_n_active_segments_w_labels(self):
        sc = segment_container.SegmentContainer("fake_audio_path")
        n_segments = 5
        active_segment_ind = [2, 3]
        for i in range(n_segments):
            sc.segments.append(segment.Segment(i * 1, (i + 1) * 1))
            if i in active_segment_ind:
                sc.segments[-1].activity = True
        assert sc.n_active_segments_with_label(segment.common_labels.unknown) == len(active_segment_ind)

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
        assert np.abs(sc.segments[0].duration - TEST_SEGMENT_DURATIONS[0]) < 1e-03
        assert np.abs(sc.segments[1].duration - TEST_SEGMENT_DURATIONS[1]) < 1e-03
        assert np.abs(sc.segments[2].duration - TEST_SEGMENT_DURATIONS[2]) < 1e-03
        assert np.abs(sc.segments[3].duration - TEST_SEGMENT_DURATIONS[3]) < 1e-03

    def test_split_data_duration(self):
        file_duration = 12.5
        seg_duration = 0.4
        seg_overlap = 0.3
        segments = segment_container.split_data(file_duration, seg_duration,
                seg_overlap)
        assert np.all(np.isclose(np.asarray([s.duration for s in segments]),
                                 seg_duration))

    def test_split_data_n_segments(self):
        file_duration = 12.5
        seg_duration = 0.4
        seg_overlap = 0.3
        segments = segment_container.split_data(file_duration, seg_duration,
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

    def test_gen_minibatches_count(self):
        sample_rate = 22050
        seg_duration = 0.1
        seg_overlap = 0.5

        parser = CSVLabelParser(TEST_CSVLABEL_PATH)
        ac_ext = AudioChunkExtractor(DATA_PATH, sample_rate)
        sf_pro = SegmentFeatureProcessor([ac_ext])
        sc_gen = SegmentContainerGenerator(
                DATA_PATH,
                sf_pro,
                label_parser=parser,
                seg_duration=seg_duration,
                seg_overlap=seg_overlap)
        
        id0132_data, sr = sf.read(os.path.join(*TEST_AUDIO_PATH_TUPLE_1))
        id1238_data, sr = sf.read(os.path.join(*TEST_AUDIO_PATH_TUPLE_2))

        n_epochs = 3
        classes = ["bird_c", "bird_d"]
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

        mb_gen = MiniBatchGen(classes,
                "audio_chunk",
                batch_size,
                1,
                n_time_bins)

        for i in range(n_epochs):
            sc_gen.reset()
            mb_gen_e = mb_gen.execute(sc_gen,
                    with_targets=True)
            count = 0
            for data, target in mb_gen_e:
                if count < id0132_n_minibatches:
                    assert np.all(target==classes.index("bird_c"))
                elif count == id0132_n_minibatches:
                    assert (len(np.where(target==classes.index("bird_c"))[0]) ==
                            id0132_n_chunks % batch_size)
                    assert (len(np.where(target==classes.index("bird_d"))[0]) ==
                            batch_size - (id0132_n_chunks % batch_size))
                else:
                    assert np.all(target==classes.index("bird_d"))
                count += 1

            assert count == n_minibatches

    def test_gen_minibatches_1d(self):
        sample_rate = 22050
        seg_duration = 0.1
        seg_overlap = 0.5
        seg_size = int(seg_duration * sample_rate)

        parser = CSVLabelParser(TEST_CSVLABEL_PATH)
        ac_ext = AudioChunkExtractor(DATA_PATH, sample_rate)
        sf_pro = SegmentFeatureProcessor([ac_ext])
        sc_gen = SegmentContainerGenerator(
                DATA_PATH,
                sf_pro,
                label_parser=parser,
                seg_duration=seg_duration,
                seg_overlap=seg_overlap)
        
        id0132_data, sr = sf.read(os.path.join(*TEST_AUDIO_PATH_TUPLE_1))
        id1238_data, sr = sf.read(os.path.join(*TEST_AUDIO_PATH_TUPLE_2))

        n_epochs = 3
        classes = ["bird_c", "bird_d"]
        batch_size = 10

        id0132_n_chunks = utils.get_n_overlapping_chunks(
                len(id0132_data),
                seg_size,
                seg_overlap)
        id1238_n_chunks = utils.get_n_overlapping_chunks(
                len(id1238_data),
                sample_rate * seg_duration,
                seg_overlap)

        id0132_n_minibatches = int(id0132_n_chunks / batch_size)
        id0132_n_chunks_in_common_minibatch = id0132_n_chunks % batch_size
        id1238_n_chunks_in_common_minibatch = \
                batch_size - id0132_n_chunks_in_common_minibatch
        id1238_n_minibatches = int((id1238_n_chunks -
            id1238_n_chunks_in_common_minibatch) / batch_size)

        id0132_minibatches = np.zeros((id0132_n_minibatches, batch_size,
            seg_size))
        start_time = 0
        for i in range(id0132_n_minibatches):
            for j in range(batch_size):
                start_ind = int(start_time*sample_rate)
                end_ind = start_ind + seg_size
                id0132_minibatches[i, j, :] = id0132_data[start_ind:end_ind]
                start_time += seg_duration * seg_overlap

        common_minibatch = np.zeros((batch_size, seg_size))
        for i in range(id0132_n_chunks_in_common_minibatch):
            start_ind = int(start_time*sample_rate)
            end_ind = start_ind + seg_size
            common_minibatch[i] = id0132_data[start_ind:end_ind]
            start_time += seg_duration * seg_overlap
        start_time = 0
        for i in range(batch_size - id0132_n_chunks_in_common_minibatch):
            start_ind = int(start_time*sample_rate)
            end_ind = start_ind + seg_size
            common_minibatch[i+id0132_n_chunks_in_common_minibatch] = \
                    id1238_data[start_ind:end_ind]
            start_time += seg_duration * seg_overlap
        
        id1238_minibatches = np.zeros((id1238_n_minibatches, batch_size,
            seg_size))
        for i in range(id1238_n_minibatches):
            for j in range(batch_size):
                start_ind = int(start_time*sample_rate)
                end_ind = start_ind + seg_size
                id1238_minibatches[i, j, :] = id1238_data[start_ind:end_ind]
                start_time += seg_duration * seg_overlap
        
        mb_gen = MiniBatchGen(classes,
                "audio_chunk",
                batch_size,
                1,
                seg_size)

        for i in range(n_epochs):
            sc_gen.reset()
            mb_gen_e = mb_gen.execute(sc_gen,
                    with_targets=True)
            count = 0
            for data, target in mb_gen_e:
                data = data.reshape((10, 2205))
                if count < id0132_n_minibatches:
                    assert np.all(data == id0132_minibatches[count])
                elif count == id0132_n_minibatches:
                    assert np.all(data == common_minibatch)
                else:
                    assert np.all(data ==
                            id1238_minibatches[count-id0132_n_minibatches-1])
                count += 1

    def test_gen_minibatches_2d(self):
        sample_rate = 22050
        win_size = 256
        hop_size = 128
        seg_duration = 0.1
        seg_overlap = 0.5
        seg_size = int(seg_duration * sample_rate)
        n_epochs = 3
        classes = ["bird_c", "bird_d"]
        batch_size = 10
        n_time_bins = int(seg_size / hop_size)

        parser = CSVLabelParser(TEST_CSVLABEL_PATH)

        # libdyni mel extractor
        n_mels = 32
        min_freq = 0
        max_freq = 11025
        af_gen = AudioFrameGen(win_size, hop_size, Window.rect)
        mel_ext = MelSpectrumExtractor(
                sample_rate=sample_rate,
                fft_size=win_size,
                n_mels=n_mels,
                min_freq=min_freq,
                max_freq=max_freq,
                log_amp=False)
        ff_pro = FrameFeatureProcessor(af_gen, [mel_ext])
        ffc_ext = FrameFeatureChunkExtractor("mel_spectrum")
        sf_pro = SegmentFeatureProcessor(
                [ffc_ext],
                ff_pro=ff_pro,
                audio_root=DATA_PATH)

        sc_gen = SegmentContainerGenerator(
                DATA_PATH,
                sf_pro,
                label_parser=parser,
                seg_duration=seg_duration,
                seg_overlap=seg_overlap)

        # librosa mel extractor
        id0132_data, sr = sf.read(os.path.join(*TEST_AUDIO_PATH_TUPLE_1))
        id0132_mel_librosa = []
        start_ind = 0
        while start_ind + win_size < len(id0132_data):
            spec = np.abs(np.fft.rfft(id0132_data[start_ind:start_ind+win_size])) ** 2
            id0132_mel_librosa.append(
                    melspectrogram(sr=sample_rate, S=spec, n_fft=win_size,
                        hop_length=hop_size, n_mels=n_mels, fmin=min_freq, fmax=max_freq))
            start_ind += hop_size
        id1238_data, sr = sf.read(os.path.join(*TEST_AUDIO_PATH_TUPLE_2))
        id1238_mel_librosa = []
        start_ind = 0
        while start_ind + win_size < len(id1238_data):
            spec = np.abs(np.fft.rfft(id1238_data[start_ind:start_ind+win_size])) ** 2
            id1238_mel_librosa.append(
                    melspectrogram(sr=sample_rate, S=spec, n_fft=win_size,
                        hop_length=hop_size, n_mels=n_mels, fmin=min_freq, fmax=max_freq))
            start_ind += hop_size

        # create segment data

        id0132_chunks = []
        start_time = 0
        start_ind = 0
        while start_ind + n_time_bins < len(id0132_mel_librosa):
            id0132_chunks.append(id0132_mel_librosa[start_ind:start_ind+n_time_bins])
            start_time += seg_duration * (1 - seg_overlap)
            start_ind = int(start_time * sample_rate / hop_size)
        id0132_n_chunks = len(id0132_chunks)
        
        id1238_chunks = []
        start_time = 0
        start_ind = 0
        while start_ind + n_time_bins < len(id1238_mel_librosa):
            id1238_chunks.append(id1238_mel_librosa[start_ind:start_ind+n_time_bins])
            start_time += seg_duration * (1 - seg_overlap)
            start_ind = int(start_time * sample_rate / hop_size)
        id1238_n_chunks = len(id1238_chunks)

        # create minibatches

        id0132_n_minibatches = id0132_n_chunks // batch_size
        id0132_minibatches = np.zeros((
            id0132_n_minibatches, batch_size, n_time_bins, n_mels))
        for i in range(id0132_n_minibatches):
            for j in range(batch_size):
                id0132_minibatches[i, j] = id0132_chunks[i*batch_size+j]

        common_minibatch = np.zeros((batch_size, n_time_bins, n_mels))
        id0132_n_chunks_in_common_minibatch = id0132_n_chunks - id0132_n_minibatches * batch_size
        for i, chunks in enumerate(id0132_chunks[-id0132_n_chunks_in_common_minibatch:]):
            common_minibatch[i] = chunks
        id1238_n_chunks_in_common_minibatch = batch_size - id0132_n_chunks_in_common_minibatch
        for i in range(id1238_n_chunks_in_common_minibatch):
            common_minibatch[id0132_n_chunks_in_common_minibatch + i] = id1238_chunks[i]

        id1238_n_minibatches = (id1238_n_chunks - id1238_n_chunks_in_common_minibatch) // batch_size
        id1238_minibatches = np.zeros((
            id1238_n_minibatches, batch_size, n_time_bins, n_mels))
        for i in range(id1238_n_minibatches):
            for j in range(batch_size):
                id1238_minibatches[i, j] = id1238_chunks[id1238_n_chunks_in_common_minibatch + i*batch_size + j]

        # and now compare

        for i in range(n_epochs):
            sc_gen.reset()
            mb_gen = gen_minibatches(
                    sc_gen,
                    classes,
                    batch_size,
                    n_mels,
                    n_time_bins,
                    "mel_spectrum")

            count = 0
            for data, target in mb_gen:
                data = np.swapaxes(data.reshape((10, n_mels, n_time_bins)), -2,
                        -1)
                if count < id0132_n_minibatches:
                    assert np.allclose(data, id0132_minibatches[count])
                elif count == id0132_n_minibatches:
                    assert np.allclose(data, common_minibatch)
                else:
                    assert np.allclose(data,
                            id1238_minibatches[count-id0132_n_minibatches-1])
                count += 1

