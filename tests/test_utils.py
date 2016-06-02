import os
import pytest

import numpy as np

from libdyni.utils.exceptions import ParameterError
from libdyni.utils import segment
from libdyni.utils import segment_container
from libdyni.utils import utils

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

TEST_AUDIO_PATH_TUPLE = (DATA_PATH, "ID0132.wav")
TEST_SEG_PATH_TUPLE = (DATA_PATH, "ID0132.seg")
TEST_DURATION = 15.45
TEST_N_SEGMENTS = 4
TEST_FIRST_SEGMENT_DURATION = 0.79


class TestSegment:

    def test_working_case(self):
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
            sc = segment_container.create_segment_containers_from_audio_file(
                os.path.join(*TEST_AUDIO_PATH_TUPLE))

    def test_create_segment_containers_from_audio_file_n_segment(self):
        sc = segment_container.create_segment_containers_from_audio_file(
            TEST_AUDIO_PATH_TUPLE)
        assert sc.n_segments == 1

    def test_create_segment_containers_from_audio_file_segment_duration(self):
        sc = segment_container.create_segment_containers_from_audio_file(
            TEST_AUDIO_PATH_TUPLE)
        assert np.abs(sc.segments[0].duration - TEST_DURATION) < 1e-03

    def test_create_segment_containers_from_seg_file_tuple(self):
        with pytest.raises(TypeError):
            sc = segment_container.create_segment_containers_from_seg_file(
                os.path.join(*TEST_SEG_PATH_TUPLE))

    def test_create_segment_containers_from_seg_file_n_segment(self):
        sc = segment_container.create_segment_containers_from_seg_file(
            TEST_SEG_PATH_TUPLE)
        assert sc.n_segments == TEST_N_SEGMENTS

    def test_create_segment_containers_from_seg_file_segment_duration(self):
        sc = segment_container.create_segment_containers_from_seg_file(
            TEST_SEG_PATH_TUPLE)
        assert np.abs(sc.segments[0].duration - TEST_FIRST_SEGMENT_DURATION) < 1e-03

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
