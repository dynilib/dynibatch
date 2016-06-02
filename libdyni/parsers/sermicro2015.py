import os


def get_label(audio_path):
    return os.path.basename(audio_path).split("_")[0]
