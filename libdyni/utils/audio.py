import soundfile as sf

def info(path):
    with sf.SoundFile(path) as f:
        return f._info

