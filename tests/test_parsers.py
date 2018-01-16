#The MIT License
#
#Copyright (c) 2017 DYNI machine learning & bioacoustics team - Univ. Toulon
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of
#this software and associated documentation files (the "Software"), to deal in
#the Software without restriction, including without limitation the rights to
#use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
#the Software, and to permit persons to whom the Software is furnished to do so,
#subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
#FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
#IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
#CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import os
import pytest

from dynibatch.parsers.label_parsers import CSVFileLabelParser, CSVSegmentLabelParser
from dynibatch.utils.segment import CommonLabels


DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

TEST_FILE2LABEL_PATH = os.path.join(DATA_PATH, "file2label.csv")
TEST_LABEL_PATH = os.path.join(DATA_PATH, "labels.txt")


class TestCSVFileLabelParser:

    def test_init(self):
        try:
            CSVFileLabelParser(TEST_FILE2LABEL_PATH, label_file=TEST_LABEL_PATH)
        except Exception as e:
            pytest.fail("Unexpected Error: {}".format(e))

    def test_get_label(self):
        parser = CSVFileLabelParser(TEST_FILE2LABEL_PATH, label_file=TEST_LABEL_PATH)
        assert parser.get_label("dataset1/ID0131.wav") == 2

    def test_get_labels(self):
        parser = CSVFileLabelParser(TEST_FILE2LABEL_PATH, label_file=TEST_LABEL_PATH)
        classes = parser.get_labels()
        assert sorted(classes.keys()) == [2, 3]
        assert sorted(classes.values()) == ["bird_b", "bird_c"]


class TestCSVSegmentLabelParser:

    def test_init(self):
        try:
            CSVSegmentLabelParser(DATA_PATH, TEST_LABEL_PATH)
        except Exception as e:
            pytest.fail("Unexpected Error: {}".format(e))


    def test_get_labels(self):
        parser = CSVSegmentLabelParser(DATA_PATH, TEST_LABEL_PATH)
        classes = parser.get_labels()
        assert sorted(classes.values()) == ["bird_b", "bird_c"]

    def test_get_label(self):
        parser = CSVSegmentLabelParser(DATA_PATH, TEST_LABEL_PATH,
                                       separator=",")
        classes = parser.get_labels()
        assert list(parser.get_segment_container("dataset1/ID0132.wav").labels)[0] == CommonLabels.unknown.value
        assert list(parser.get_segment_container("dataset1/ID0133.wav").labels)[0] == 3
