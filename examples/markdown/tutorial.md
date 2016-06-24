
# Tutorial

The main purpose of libdyni is to ease the creation of mini-batches (data + targets) to be fed to some machine learning algorithm.

## Data structures

### Segment

Segments are the base elements of the mini-batches. From a machine learning perspective, 1 segment = 1 example.

Audio files are split into overlapping fixed-length segments, stored in *segment containers*.

Every segment, along with its parent segment container, contains all the data needed to feed a mini-batch (label, features, whether it contains activity or not).

### Segment container

A segment container is related to an audio file (1 segment container per audio file). It contains the path of the audio file as well as a list of segments.

### Feature container

A feature container contains all the short-term features (e.g. spectral flatness, mel-spectra...), as well as their parameters, computed from a given audio file. When saved on disk, features are not directly stored in segments because that would imply a lot of duplicated data (since segments most often overlap). Instead, they are saved as feature container dumps.

### Dataset split

A dataset split describes how a dataset is split into train/validation/test sets. It is basically a dictionary with a key for each set and a list of files as its value.


## Examples

[Create segment containers and set labels](#Create-segment-containers-and-set-labels)

[Plot a frame from an audio file](#Plot-a-frame-from-an-audio-file)

### Create segment containers and set labels


```python
from libdyni.utils.segment_container import create_segment_containers_from_audio_files
from libdyni.parsers.label_parsers import CSVLabelParser


# create a segment container generator
sc_gen = create_segment_containers_from_audio_files("tests/data")

# instanciate the label parser (labels.csv contains the pairs file/label)
parser = CSVLabelParser("tests/data/labels.csv")

# now for every segment container, set and show the label
for sc in sc_gen:
    sc.labels = parser.get_label(sc.audio_path)
    print("Label for file {0}: {1}".format(sc.audio_path, sc.labels))
```

    Label for file ID0132.wav: {'bird_c'}
    Label for file ID1238.wav: {'bird_d'}


### Plot a frame from an audio file


```python
import os
import matplotlib.pyplot as plt

from libdyni.generators.audio_frame_gen import AudioFrameGen

# some matplotlib config
%matplotlib inline
%pylab inline
pylab.rcParams['figure.figsize'] = (14, 8)

# data
audio_root = "tests/data"
audio_filename = "ID0132.wav"

# create and execute an audio frame generator
win_size = 256
hop_size = 128
af_gen = AudioFrameGen(win_size, hop_size) # default window is hanning
af_gen_e = af_gen.execute(os.path.join(audio_root, "ID0132.wav"))

# get third frame, and plot
next(af_gen_e) # 1st frame
next(af_gen_e) # 2nd frame
frame = next(af_gen_e) # 3rd frame
plt.plot(frame)
```

    Populating the interactive namespace from numpy and matplotlib





    [<matplotlib.lines.Line2D at 0x7f7d13f6c1d0>]




![png](output_3_2.png)


### Generate mini-batches with mel-spectra


```python
from libdyni.generators.segment_container_gen import SegmentContainerGenerator
from libdyni.generators.audio_frame_gen import AudioFrameGen
from libdyni.features.extractors.energy import EnergyExtractor
from libdyni.features.extractors.spectral_flatness import SpectralFlatnessExtractor
from libdyni.features.extractors.mel_spectrum import MelSpectrumExtractor
from libdyni.features.frame_feature_processor import FrameFeatureProcessor
from libdyni.features.segment_feature_processor import SegmentFeatureProcessor
from libdyni.features.extractors.frame_feature_chunk import FrameFeatureChunkExtractor
from libdyni.features.extractors.activity_detection import ActivityDetection
from libdyni.parsers import label_parsers
from libdyni.utils.batch import gen_minibatches


#################
# Configuration #
#################

# audio and short-term frames config
audio_root = "tests/data"
sample_rate = 22050
win_size = 256
hop_size = 128

# mel spectra config
n_mels = 64
min_freq = 0
max_freq = sample_rate / 2

# segments config
seg_duration = 0.2
seg_overlap = 0.5

# activity detection config
energy_threshold = 0.2
spectral_flatness_threshold = 0.3

# mini-batches config
batch_size = 10
n_features = n_mels
n_time_bins = int(seg_duration * sample_rate / hop_size)

# sorted list of labels for the audio files in audio_root
# the file/label pairs are defined in label.csv
labels = sorted(["bird_c", "bird_d"])

##############
# Processing #
##############

# create a parser to get the labels from the labels.csv file
parser = label_parsers.CSVLabelParser("tests/data/labels.csv")

# create needed short-term (aka frame-based) feature extractors
en_ext = EnergyExtractor() # needed for the activity detection
sf_ext = SpectralFlatnessExtractor() # needed for the activity detection
mel_ext = MelSpectrumExtractor(    
    sample_rate=sample_rate,
    fft_size=win_size,
    n_mels=n_mels,
    min_freq=min_freq,
    max_freq=max_freq)

# create an audio frame generator
af_gen = AudioFrameGen(win_size=win_size, hop_size=hop_size)

# create a frame feature processor, in charge of computing all short-term features
ff_pro = FrameFeatureProcessor(
    af_gen,
    [en_ext, sf_ext, mel_ext])

# create needed segment-based feature extractors
ffc_ext = FrameFeatureChunkExtractor(mel_ext.name)
act_det = ActivityDetection(
        energy_threshold=energy_threshold,
        spectral_flatness_threshold=spectral_flatness_threshold)

# create a segment feature processor, in charge of computing all segment-based features
# (here only chunks of mel spectra sequences)
sf_pro = SegmentFeatureProcessor(
        [act_det, ffc_ext],
        ff_pro=ff_pro,
        audio_root=audio_root)

# create and start the segment container generator that will use all the objects above to generate
# for every audio files a segment container containing the list of segments with the labels,
# the mel spectra and an "activity detected" boolean attribute
sc_gen = SegmentContainerGenerator(
       audio_root,
       parser,
       sf_pro,
       seg_duration=seg_duration,
       seg_overlap=seg_overlap)
sc_gen.start()

# generate mini-batches
mb_gen = gen_minibatches(
    sc_gen,
    labels,
    batch_size,
    n_features,
    n_time_bins,
    mel_ext.name,
    active_segment_only=True)

# get the first mini-batch
data, target = next(mb_gen)
```

    /home/jul/Development/miniconda/envs/libdyni/lib/python3.5/site-packages/librosa/core/audio.py:37: UserWarning: Could not import scikits.samplerate. Falling back to scipy.signal
      warnings.warn('Could not import scikits.samplerate. '



```python
# Show the label indices of the first mini-batch
print("Label indices:\n{}".format(target))
```

    Label indices:
    [0 0 0 0 0 0 0 0 0 0]



```python
# and the data
print("Mel spectra (truncated):\n{}".format(data))
```

    Mel spectra (truncated):
    [[[[-77.88076782 -66.65984344 -85.78934479 ..., -61.77670288 -69.87345886
        -66.31863403]
       [-74.55519867 -63.33428192 -82.46378326 ..., -58.45114136 -66.54789734
        -62.9930687 ]
       [-66.03651428 -51.91387939 -58.83434677 ..., -49.6537323  -54.32802963
        -62.22982788]
       ..., 
       [-52.81373978 -56.42927933 -57.59656143 ..., -45.98276901 -46.00500107
        -45.56991959]
       [-54.35548019 -55.33501053 -53.44735718 ..., -52.01996231 -51.14955139
        -50.94314575]
       [-61.97282028 -56.9071846  -58.29795074 ..., -59.93020248 -59.93471146
        -54.18180847]]]
    
    
     [[[-75.68052673 -72.4744339  -95.34684753 ..., -92.85736847 -68.09340668
        -61.43295288]
       [-72.35495758 -69.14886475 -92.02128601 ..., -89.53180695 -64.76783752
        -58.10738754]
       [-59.55133438 -55.01808548 -62.05028534 ..., -55.01891327 -55.13725662
        -52.58815765]
       ..., 
       [-53.12981415 -50.01371765 -51.11210632 ..., -54.59425354 -51.61140442
        -54.05944443]
       [-54.07550812 -53.38811111 -53.34747696 ..., -56.17362595 -52.94422913
        -59.890522  ]
       [-58.15501022 -57.97169113 -56.77929306 ..., -61.40216446 -52.91544342
        -63.99854279]]]
    
    
     [[[-67.12281799 -68.72060394 -72.96092987 ..., -87.88312531 -68.47286987
        -74.69589996]
       [-63.79724884 -65.39504242 -69.63536835 ..., -84.55755615 -65.14730072
        -71.37033844]
       [-51.50327682 -58.38234711 -60.64006042 ..., -52.32159042 -53.42242432
        -65.59535217]
       ..., 
       [-50.74246597 -44.23286438 -45.24972534 ..., -49.06516266 -47.90545273
        -51.40767288]
       [-53.79285049 -50.02662277 -52.67823029 ..., -50.08366776 -52.70608902
        -49.46236801]
       [-62.03483963 -53.43784714 -57.04169464 ..., -54.90446472 -56.12475586
        -54.917099  ]]]
    
    
     ..., 
     [[[-68.06482697 -70.6432724  -68.73229218 ..., -80.74584961 -66.31039429
        -67.38722992]
       [-64.73926544 -67.31771088 -65.40673065 ..., -77.42028046 -62.98482895
        -64.0616684 ]
       [-56.84818268 -53.10140228 -49.20680237 ..., -53.4901619  -62.23062897
        -55.5881424 ]
       ..., 
       [-53.16926575 -49.32261658 -52.52849579 ..., -53.68796539 -53.36948776
        -55.63058853]
       [-52.54328918 -50.94367981 -51.08142471 ..., -53.9724617  -53.37283707
        -54.73107529]
       [-61.43606186 -54.11021042 -56.64054108 ..., -54.33579636 -56.93286896
        -55.42055893]]]
    
    
     [[[-70.57924652 -73.09832764 -70.07554626 ..., -77.24311829 -64.16478729
        -70.66811371]
       [-67.253685   -69.77275848 -66.74997711 ..., -73.91755676 -60.83922195
        -67.34255219]
       [-58.06739426 -63.24688339 -62.54263687 ..., -50.91062546 -53.3068924
        -54.68761826]
       ..., 
       [-52.42127228 -48.13520813 -53.96286392 ..., -48.01916504 -48.5043335
        -51.35004425]
       [-50.84687424 -51.52099991 -58.50524521 ..., -49.02523422 -47.29510117
        -51.22759628]
       [-58.3094368  -57.08455658 -58.83781815 ..., -50.5095787  -56.91412735
        -51.96370316]]]
    
    
     [[[-73.56750488 -67.71328735 -72.21214294 ..., -76.56441498 -72.44521332
        -64.8015213 ]
       [-70.24194336 -64.3877182  -68.88658142 ..., -73.23885345 -69.11964417
        -61.47595596]
       [-51.82940674 -54.89487076 -51.79399872 ..., -53.71609497 -48.68716049
        -48.87018204]
       ..., 
       [-48.73300552 -44.48386383 -50.62590027 ..., -49.68765259 -53.97488785
        -54.15042496]
       [-50.96985245 -49.78147888 -53.98550034 ..., -54.57501221 -51.60131073
        -57.00195312]
       [-54.23483276 -56.22404099 -57.84348679 ..., -57.31592941 -54.84722137
        -59.9726181 ]]]]



```python

```
