import os
import pytest

from libdyni.parsers.label_parsers import CSVLabelParser


DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

TEST_FILE2LABEL_PATH = os.path.join(DATA_PATH, "file2label.csv")


class TestCSVLabelParser:

    def test_init(self):
        try:
            CSVLabelParser(TEST_FILE2LABEL_PATH)
        except Exception as e:
            pytest.fail("Unexpected Error: {}".format(e))

    def test_get_label(self):
        parser = CSVLabelParser(TEST_FILE2LABEL_PATH)
        classes = parser.get_labels()
        assert parser.get_label("ID0131") == classes.index("bird_b")
