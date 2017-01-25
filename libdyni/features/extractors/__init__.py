import inspect
import importlib

from libdyni.utils import exceptions


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
        raise exceptions.LibdyniError("Extractor module {} must contain \
                exactly one class.".format(module))

    # configure extractor and return
    return clsmembers[0].from_config_dict(audio_frame_config, feature_config)

