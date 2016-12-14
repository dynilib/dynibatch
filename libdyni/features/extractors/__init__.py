import inspect
import importlib

from libdyni.utils import exceptions


def factory(name, audio_frame_config={}, config={}):
    """Create an instance of the extractor,
    as defined by its name and configured by audio_frame_config and config"""

    # get module from name
    module = importlib.import_module("." + name, package=str(__package__))

    # get extractor class, making sure there is only one (because we
    # pick the first one)
    clsmembers = inspect.getmembers(module, inspect.isclass)
    if not len(clsmembers) == 1:
        raise exceptions.LibdyniError("Extractor module {} must contain
                exactly one class.".format(module))

    _, cls = clsmembers[0]

    # configure extractor and return
    return cls.from_config_dict(audio_frame_config, config)

