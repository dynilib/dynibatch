# Dynibatch

Dynibatch is a Python library dedicated to providing mini-batches of audio data to machine learning algorithms.

It has been designed to deal with the following issues:

* **Reproducibility**: given a dataset (data + train/valid/test split + labels) and some parameters (e.g. segment size and overlap, audio feature to be used...), an experiment should be easily reproducible. Dynibatch allows to keep all this information in dedicated objects and a configuration file.

* **Big data**: because some datasets are huge, Dynibatch keeps a low memory footprint by generating mini-batches on the fly. To avoid recomputing at every epoch the data needed to generate the mini-batches, it can be cached in the disk.

* **Label management**: labels are typically provided either *per file* (i.e. one label for the whole audio file) or *per chunk* (i.e. one label for an audio chunk, delimited by a start time and an end time). In either case, Dynibatch automatically maps the labels provided with the dataset to *one label per segment*, where a segment is one fixed-size observation (see the tutorial for more details TODO add link). Labels are not mandatory, so that unsupervised algorithms can be run.

* **Usability**: with a given config file, generating mini-batches is as easy as

        mb_gen = MiniBatchGen.from_config(config)
        mb_gen.start()
        mb_gen_e = mb_gen.execute(with_targets=True)

        # get the first mini-batch
        data, targets = next(mb_gen_e)

More details and examples can be found in the tutorial. 

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