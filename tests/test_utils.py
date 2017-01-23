import os
import random
import pytest

import numpy as np

from libdyni.utils.exceptions import ParameterError
from libdyni.utils import segment
from libdyni.utils import segment_container
from libdyni.utils import feature_container
from libdyni.utils import datasplit_utils
from libdyni.utils import utils
from libdyni.parsers import label_parsers


DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

TEST_AUDIO_PATH_TUPLE_1 = (DATA_PATH, "ID0132.wav")
TEST_AUDIO_PATH_TUPLE_2 = (DATA_PATH, "ID1238.wav")
TEST_SEG_PATH_TUPLE_1 = (DATA_PATH, "ID0132.seg")
TEST_SEG_PATH_TUPLE_2 = (DATA_PATH, "ID0133.seg")
TEST_DURATION = 15.45
TEST_N_SEGMENTS = 4
TEST_FIRST_SEGMENT_DURATION = 0.79
SEGMENT_CONTAINER_LISTS_TO_GENERATE = 100
TEST_FILE2LABEL_PATH = os.path.join(DATA_PATH, "file2label.csv")
TEST_LABELS_PATH = os.path.join(DATA_PATH, "labels.txt")

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
        segment.set_segment_labels(segment_from_list,
                                   segment_to_list,
                                   overlap_ratio=0.5)
        assert(segment_to_list[0].label == "a" and
               segment_to_list[1].label == "a" and
               segment_to_list[2].label == "b" and
               segment_to_list[3].label == segment.CommonLabels.unknown.value)

    def test_set_segment_labels_overlap(self):
        segment_from_list = []
        segment_from_list.append(segment.Segment(0, 1, "a"))
        segment_to_list = []
        segment_to_list.append(segment.Segment(0.4, 1.5))
        segment_to_list.append(segment.Segment(0.6, 1.5))
        segment.set_segment_labels(segment_from_list,
                                   segment_to_list,
                                   overlap_ratio=0.5)
        assert(segment_to_list[0].label == "a" and
               segment_to_list[1].label == segment.CommonLabels.unknown.value)


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
        assert sc.n_segments_with_label(segment.CommonLabels.unknown.value) == n_segments

    def test_n_active_segments_w_labels(self):
        sc = segment_container.SegmentContainer("fake_audio_path")
        n_segments = 5
        active_segment_ind = [2, 3]
        for i in range(n_segments):
            sc.segments.append(segment.Segment(i * 1, (i + 1) * 1))
            if i in active_segment_ind:
                sc.segments[-1].activity = True
        assert sc.n_active_segments_with_label(segment.CommonLabels.unknown.value) == len(active_segment_ind)

    def test_create_random_segment_containers(self):
        sc_ref = segment_container.create_segment_containers_from_audio_files(
            DATA_PATH,
            randomize=True,
            label_parser=None)

        sc_generated = []
        for _ in range(SEGMENT_CONTAINER_LISTS_TO_GENERATE):
            sc_generated.append(segment_container.create_segment_containers_from_audio_files(
                DATA_PATH,
                randomize=True,
                label_parser=None))

        sc_ref = list(sc_ref)
        list_equals = 0
        for sc_try in sc_generated:
            list_equals += sc_ref == list(sc_try)

        assert list_equals < SEGMENT_CONTAINER_LISTS_TO_GENERATE


    def test_create_stratified_segment_containers(self):
        label_parser = label_parsers.CSVFileLabelParser(TEST_FILE2LABEL_PATH)

        sc_ref = segment_container.create_segment_containers_from_audio_files(
            DATA_PATH,
            randomize=False,
            label_parser=label_parser)

        sc_generated = []
        for _ in range(SEGMENT_CONTAINER_LISTS_TO_GENERATE):
            sc_generated.append(segment_container.create_segment_containers_from_audio_files(
                DATA_PATH,
                randomize=False,
                label_parser=label_parser))

        sc_ref = list(sc_ref)
        list_equals = 0
        for sc_try in sc_generated:
            list_equals += sc_ref == list(sc_try)

        assert list_equals == SEGMENT_CONTAINER_LISTS_TO_GENERATE


    def test_create_segment_container_from_audio_file_tuple(self):
        with pytest.raises(TypeError):
            segment_container.create_segment_container_from_audio_file(
                    os.path.join(*TEST_AUDIO_PATH_TUPLE_1))

    def test_create_segment_container_from_audio_file_n_segment(self):
        sc = segment_container.create_segment_container_from_audio_file(
            TEST_AUDIO_PATH_TUPLE_1)
        assert sc.n_segments == 1

    def test_create_segment_container_from_audio_file_segment_duration(self):
        sc = segment_container.create_segment_container_from_audio_file(
            TEST_AUDIO_PATH_TUPLE_1)
        assert np.abs(sc.segments[0].duration - TEST_DURATION) < 1e-03

    def test_create_segment_container_from_seg_file_tuple(self):
        with open(TEST_LABELS_PATH, "r") as f:
            labels = [l.strip() for l in f.readlines() if l.strip()]
            with pytest.raises(TypeError):
                segment_container.create_segment_container_from_seg_file(
                        os.path.join(*TEST_SEG_PATH_TUPLE_1), labels)

    def test_create_segment_container_from_seg_file_n_segment(self):
        with open(TEST_LABELS_PATH, "r") as f:
            labels = [l.strip() for l in f.readlines() if l.strip()]
            sc = segment_container.create_segment_container_from_seg_file(
                TEST_SEG_PATH_TUPLE_1, labels)
            assert sc.n_segments == TEST_N_SEGMENTS

    def test_create_segment_container_from_seg_file_segment_duration(self):
        with open(TEST_LABELS_PATH, "r") as f:
            labels = [l.strip() for l in f.readlines() if l.strip()]
            sc = segment_container.create_segment_container_from_seg_file(
                TEST_SEG_PATH_TUPLE_1, labels)
            assert np.abs(sc.segments[0].duration - TEST_FIRST_SEGMENT_DURATION) < 1e-03
    
    def test_create_segment_container_from_seg_file_labels(self):
        with open(TEST_LABELS_PATH, "r") as f:
            labels = [l.strip() for l in f.readlines() if l.strip()]
            sc_1 = segment_container.create_segment_container_from_seg_file(
                TEST_SEG_PATH_TUPLE_1, labels)
            sc_2 = segment_container.create_segment_container_from_seg_file(
                TEST_SEG_PATH_TUPLE_2, labels)
            assert sc_1.segments[0].label == segment.CommonLabels.unknown.value
            assert sc_2.segments[0].label == labels.index("bird_c")

    def test_create_fixed_duration_segments_duration(self):
        file_duration = 12.5
        seg_duration = 0.4
        seg_overlap = 0.3
        segments = segment_container.create_fixed_duration_segments(file_duration,
                                                                    seg_duration,
                                                                    seg_overlap)
        assert np.all(np.isclose(np.asarray([s.duration for s in segments]),
                                 seg_duration))

    def test_create_fixed_duration_segments_n_segments(self):
        file_duration = 12.5
        seg_duration = 0.4
        seg_overlap = 0.3
        segments = segment_container.create_fixed_duration_segments(file_duration,
                                                                    seg_duration,
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
            feature_container.FeatureContainer("fake_audio_path",
                                               22050,
                                               256,
                                               128)
        except:
            pytest.fail("Unexpected Error")

    def test_features_ok(self):
        features = ["feat1", "feat2"]
        configs = ["config1", "config2"]
        fc = feature_container.FeatureContainer("fake_audio_path",
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
        fc = feature_container.FeatureContainer("fake_audio_path",
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
        fc = feature_container.FeatureContainer("fake_audio_path",
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
        fc = feature_container.FeatureContainer("fake_audio_path",
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
        fc = feature_container.FeatureContainer("fake_audio_path",
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

        assert utils.get_n_overlapping_chunks(file_duration,
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
            datasplit_utils.create_datasplit(train_set,
                                             validation_set,
                                             test_set,
                                             name="fake_datasplit")
        except Exception as e:
            pytest.fail("Unexpected Error: {}".format(e))

    def test_create_datasplit_count(self, file_list):
        file_set = set(file_list)
        train_set = set(random.sample(file_set, 700))
        validation_set = set(random.sample(file_set-train_set, 100))
        test_set = set(random.sample(file_set-train_set-validation_set, 200))
        ds = datasplit_utils.create_datasplit(train_set,
                                              validation_set,
                                              test_set,
                                              name="fake_datasplit")
        assert (len(ds["sets"]["train"]) == 700 and
                len(ds["sets"]["validation"]) == 100 and
                len(ds["sets"]["test"]) == 200)

    def test_create_random_datasplit_init(self, sc_list):
        try:
            datasplit_utils.create_random_datasplit(sc_list,
                                                    train_ratio=0.7,
                                                    validation_ratio=0.1,
                                                    test_ratio=0.2)
        except Exception as e:
            pytest.fail("Unexpected Error: {}".format(e))

    def test_create_random_datasplit_set_dont_sumup_to_one(self, sc_list):
        with pytest.raises(ParameterError):
            datasplit_utils.create_random_datasplit(sc_list,
                                                    train_ratio=0.8,
                                                    validation_ratio=0.1,
                                                    test_ratio=0.2)

    def test_create_random_datasplit_count(self, sc_list):
        ds = datasplit_utils.create_random_datasplit(sc_list,
                                                     train_ratio=0.7,
                                                     validation_ratio=0.1,
                                                     test_ratio=0.2)
        assert (len(ds["sets"]["train"]) == 700 and
                len(ds["sets"]["validation"]) == 100 and
                len(ds["sets"]["test"]) == 200)

    def test_datasplit_stats(self):
        # TODO (jul)
        pass
