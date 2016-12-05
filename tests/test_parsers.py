import pytest
import os

from libdyni.utils.label_parsers import CSVLabelParser
from libdyni.utils.config_parser import parse_config_file

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


class TestConfigParser:
    """
        Test module utils/config_parser
    """

    def test_init(self):
        try:
            parse_config_file("tests/config/config_test.json")
        except Exception as e:
            pytest.fail("Unexpected Error: {}".format(e))

    def test_get_minibatch(self):
        mb_gen, sc_gen = parse_config_file("tests/config/config_test.json")
        try:
            mb_gen.execute(sc_gen)
        except Exception as e:
            pytest.fail("Unexpected Error: {}".format(e))
