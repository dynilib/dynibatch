import os
import pytest

from libdyni.parsers.label_parsers import CSVFileLabelParser, CSVSegmentLabelParser
from libdyni.utils.segment import CommonLabels


DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

TEST_FILE2LABEL_PATH = os.path.join(DATA_PATH, "file2label.csv")
TEST_LABEL_PATH = os.path.join(DATA_PATH, "labels.txt")


class TestCSVFileLabelParser:

    def test_init(self):
        try:
            CSVFileLabelParser(TEST_FILE2LABEL_PATH)
        except Exception as e:
            pytest.fail("Unexpected Error: {}".format(e))

    def test_get_label(self):
        parser = CSVFileLabelParser(TEST_FILE2LABEL_PATH)
        classes = parser.get_labels()
        assert parser.get_label("dataset1/ID0131.wav") == classes.index("bird_b")
    
    def test_get_labels(self):
        parser = CSVFileLabelParser(TEST_FILE2LABEL_PATH)
        classes = parser.get_labels()
        assert classes == ["bird_a", "bird_b", "bird_c", "bird_d"]


class TestCSVSegmentLabelParser:

    def test_init(self):
        try:
            CSVSegmentLabelParser(DATA_PATH, TEST_LABEL_PATH)
        except Exception as e:
            pytest.fail("Unexpected Error: {}".format(e))

    
    def test_get_labels(self):
        parser = CSVSegmentLabelParser(DATA_PATH, TEST_LABEL_PATH)
        classes = parser.get_labels()
        assert classes == ["bird_b", "bird_c"]
    
    def test_get_label(self):
        parser = CSVSegmentLabelParser(DATA_PATH, TEST_LABEL_PATH,
                seg_file_separator="\t")
        classes = parser.get_labels()
        assert list(parser.get_segment_container("dataset1/ID0132.wav").labels)[0] == CommonLabels.unknown.value
        assert list(parser.get_segment_container("dataset1/ID0133.wav").labels)[0] == classes.index("bird_c")
