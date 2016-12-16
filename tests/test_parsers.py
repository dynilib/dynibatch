import os
import pytest

from libdyni.parsers.label_parsers import CSVLabelParser


DATA_PATH = os.path.join(os.path.dirname(__file__), "data/reduced_set")

TEST_CSVLABEL_PATH = os.path.join(DATA_PATH, "labels.csv")


class TestCSVLabelParser:

    def test_init(self):
        try:
            CSVLabelParser(TEST_CSVLABEL_PATH)
        except Exception as e:
            pytest.fail("Unexpected Error: {}".format(e))

    def test_get_label(self):
        parser = CSVLabelParser(TEST_CSVLABEL_PATH)
        classes = parser.get_labels()
        assert parser.get_label("ID0131") == classes["bird_b"]
