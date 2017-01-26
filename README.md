# Dynibatch

Dynibatch is a library, written in Python, dedicated to audio data processing for machine learning tasks. One of its main objective is to ease the creation of minibatches (data + targets) to be fed to some machine learning algorithm.

## Install

These instructions have been tested on Ubuntu 14.04.

### Miniconda

Install Miniconda (a subset of Anaconda) with Python 3.5 following the instructions given in  
http://conda.pydata.org/miniconda.html.

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

TODO

## Dependencies

The list of dependencies is provided as information only, since they should all be installed during the creation of the dynibatch conda environment. 
* [NumPy](http://www.numpy.org/)
* [SciPy](https://www.scipy.org/)
* [scikit-learn](http://scikit-learn.org)
* [librosa](https://github.com/librosa/librosa)
* [PySoundFile](https://github.com/bastibe/PySoundFile)

## License

TODO
