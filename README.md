# Dynibatch

Dynibatch is a Python library dedicated to providing mini-batches of audio data to machine learning algorithms.

Typical audio datasets consist in a set of audio files and their corresponding annotation files. Every annotation file usually contains a list of annotated segments of different sizes, each described by a start time, an end time and a label. For example:

```
0.092 0.171 hh
0.170 0.240 eh
0.240 0.282 l
0.282 0.550 ow
```

Typical machine learning algorithms take as input a batch of N overlapping segments of equal shapes S and their corresponding N labels.

Dynibatch makes it easy to create those batches of constant size segments from variable size segments. With the annotated segment examples given above, a mini-batch size of 5, a segment size S of 0.05 s and a segment overlap of 50%, Dynibatch would create mini-batches defined by the constant-size segments shown below (for a given segment, the label is set to the label in the ground truth having more than 50% overlap with this segment):

Mini-batch 1

```
0.100 0.150 hh
0.125 0.175 hh
0.150 0.200 eh 
0.175 0.225 eh
0.200 0.250 eh

```

Mini-batch 2
```
0.225 0.275 l
0.250 0.300 l
0.275 0.325 ow
0.300 0.350 ow
0.325 0.375 ow
```

...


In addition, Dynibatch can fill the mini-batches with audio features (e.g. mel spectra) instead of raw audio, and reject segments with no activity detected in it.

See examples in the tutorial.


## Install

The instructions below have been tested on Ubuntu 16.04 and [Miniconda](http://conda.pydata.org/miniconda.html) + Python 3.5.

### Get Dynibatch source

```
$ git clone https://<yourusername>@bitbucket.org/jul_dyni/dynibatch.git
```

### Create conda environment

```
$ cd dynibatch
$ conda env create -f dynibatch.yml
```

### Activate conda environment

```
$ source activate dynibatch
```

### Add dynibatch to your PYTHONPATH

```
$ export PYTHONPATH=$PYTHONPATH:<path to dynibatch>
```

### Test

Make sure all the tests pass.

```
$ py.test tests
```

## Examples

See tutorial.

## Dependencies

The list of dependencies is provided as information only, since they should all be installed during the creation of the dynibatch conda environment.

* [NumPy](http://www.numpy.org/)
* [SciPy](https://www.scipy.org/)
* [scikit-learn](http://scikit-learn.org)
* [librosa](https://github.com/librosa/librosa)
* [PySoundFile](https://github.com/bastibe/PySoundFile)

## Contact

Julien Ricard  
Vincent Roger  
Hervé Glotin  
DYNI, LSIS, University of Toulon, France

dyni<dot>contact<at>gmail.com