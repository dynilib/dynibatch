import inspect
import importlib

from libdyni.utils import exceptions


def factory(name, audio_frame_config={}, config={}):
    """Create an instance of the activity detector,
    as defined by its name and configured by audio_frame_config and config"""

    # get module from name
    module = importlib.import_module("." + name, package=str(__package__))

    # get activity detection class, making sure there is only one (because we
    # pick the first one)
    clsmembers = inspect.getmembers(module, inspect.isclass)
    if not len(clsmembers) == 1:
        raise exceptions.LibdyniError("Activity detection module {} must contain
                exactly one class.".format(module))

    _, cls = clsmembers[0]

    # configure activity detection and return
    return cls.from_config_dict(audio_frame_config, config)
