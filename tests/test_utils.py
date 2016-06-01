import pytest

from libdyni.utils import segment
from libdyni.utils.exceptions import ParameterError


class TestSegment:
    """
    - create normal segment
    - check wrong times
    """
    
    def test_working_case(self):
        try:
            start_time = 1
            end_time = 10
            s = segment.Segment(start_time, end_time)
        except ParameterError:
            pytest.fail("Unexpected ParameterError")

    def test_negative_start_time(self):
        with pytest.raises(ParameterError):
            start_time = -1
            end_time = 10
            s = segment.Segment(start_time, end_time)

    def test_time_order(self):
        with pytest.raises(ParameterError):
            start_time = 3
            end_time = 1
            s = segment.Segment(start_time, end_time)
        
    

