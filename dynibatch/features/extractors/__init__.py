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


import inspect
import importlib

from dynibatch.utils import exceptions


def factory(name, audio_frame_config={}, feature_config={}):
    """Feature extractor factory.

    Args:
        audio_frame_config (dict): AudioFrameGen config 
        feature_config (dict): feature config

    Returns:
        an instance of the feature extractor, as defined by its name and configured
        by audio_frame_config and feature_config.
    """

    # get module from name
    module = importlib.import_module("." + name, package=str(__package__))

    # get extractor class, making sure there is only one (because we
    # pick the first one)
    clsmembers = [cls for name, cls in inspect.getmembers(module,
        inspect.isclass) if cls.__module__ == module.__name__]
    if not len(clsmembers) == 1:
        raise exceptions.DynibatchError("Extractor module {} must contain \
                exactly one class.".format(module))

    # configure extractor and return
    return clsmembers[0].from_config_dict(audio_frame_config, feature_config)

