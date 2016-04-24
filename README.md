# DeepDream Workshop

The [Bay Area Women in Machine Learning & Data Science](http://www.meetup.com/Bay-Area-Women-in-Machine-Learning-and-Data-Science/) meetup group hosted a [DeepDream Workshop](http://www.meetup.com/Bay-Area-Women-in-Machine-Learning-and-Data-Science/events/224080723/) meetup on August 25, 2015.  This file walks through the installation of the software required to create "Deep Dream" images like the one below using Google's [demo notebook](https://github.com/google/deepdream).  Currently, we have OS X install instructions (using [Homebrew](http://brew.sh/) Python) only.  

![Puppyslug](./deepdream.png "Puppyslug")

## Prerequisites

The DeepDream code uses [Caffe](http://caffe.berkeleyvision.org/) and Python, so you will first need to install these prerequisites.  Below, we will walk through installing everything you need to start deep-dreaming.  These instructions fill in the details of the official [Caffe OS X install instructions](https://github.com/BVLC/caffe/wiki/Installation-%28OSX%29).

### Python 
This section will walk through installing Python, IPython/Jupyter and pip.

If you don't already have Python installed or if you are using the default Mac Python installtion, I'd recommend installing Homebrew and then install Python via [Homebrew](http://brew.sh/).

```bash
# Homebrew
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

If you already have Homebrew installed, you may want to update:

```bash
brew update
```
If you see warnings when running `brew doctor`, it might be because you are an R user who installed the official R binary.  That's shouldn't be a problem and is exaplained in further detail [here](http://azaleasays.com/2014/08/25/homebrew-warnings-about-unbrewed-dylibs-installed-by-r/).


This will install Python 2.7 via Homebrew.  If you use [Anaconda](https://www.continuum.io/downloads) or some other system to manage your Python installation, then you can ignore this step.  Just make sure that you have the appropriate Python packages installed.

```bash
# Python 2.7
brew install python
```

Homebrew should update your `PATH` variable so that when you type `python`, it will default to the Homebrew version, but to verify, you can type the following:

```bash
which python
```
It should return `/usr/local/bin/python` if Homebrew Python is the default.



Install a handful of Python packages via pip: 

```bash
pip install -U numpy
pip install -U scipy
pip install -U jupyter  #or pip install -U ipython
```

There are a few other Python dependencies for pycaffe (Python API for Caffe), but we will install those in the section below.

### Caffe

The reference installation instructions for OS X are [here](http://caffe.berkeleyvision.org/install_osx.html), but we will step through them one-by-one in this section for clarity.  There are also some OS X installation notes on the [wiki](https://github.com/BVLC/caffe/wiki/Installation-%28OSX%29).  A high-level overview of the dependecies is as follows:

- [CUDA](https://en.wikipedia.org/wiki/CUDA) is required for GPU mode
- [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) via ATLAS, MKL, or OpenBLAS (Already installed via Accelerate framework on OS X)
- [Boost](http://www.boost.org/) >= 1.55
- [OpenCV](http://opencv.org/) >= 2.4 including 3.0
- protobuf, glog, gflags
- IO libraries hdf5, leveldb, snappy, lmdb

The first thing you may want to do is install the [OS X Command Line Tools](http://osxdaily.com/2014/02/12/install-command-line-tools-mac-os-x/).  To check that you have this installed (it will install if you don't have it), run the following command:

```bash
xcode-select --install
```


#### CUDA Support
Standard Macbook Pros do not have an NVIDIA GPU (and will therefore not be possible to use Caffe's GPU mode) so you can skip the "Install CUDA" step.  If you do have an NVIDIA GPU, then you can install the CUDA dmg or pkg file [here](https://developer.nvidia.com/cuda-downloads), or `wget` and execute the dmg file to begin a network installation:

```bash
wget http://developer.download.nvidia.com/compute/cuda/7_0/Prod/network_installers/cuda_7.0.29_mac_network.dmg
open cuda_7.0.29_mac_network.dmg
```

#### Caffe software dependencies
According to the Caffe authors, it will be best if the `LD_LIBRARY_PATH` environment variable is not set.  To check, type the following (it should return a blank line if it's not set):

```bash
echo $LD_LIBRARY_PATH
```

General Caffe dependencies can be installed via Homebrew:

```bash
brew install -vd snappy leveldb gflags glog szip lmdb
# need the homebrew science source for OpenCV and hdf5
brew tap homebrew/science
brew install hdf5 opencv
```

Since we are going to use pycaffe:

```bash
# with Python pycaffe needs dependencies built from source
brew install --build-from-source --with-python -vd protobuf
brew install --build-from-source -vd boost boost-python
```
(For the last command, there was a warning of `Warning: boost-1.58.0 already installed` since I had previously brew installed boost, but this didn't cause any issues.)

There is a note on the Caffe [wiki](https://github.com/BVLC/caffe/wiki/Installation-%28OSX%29) that warns against using boost 1.56 and 1.58.  I did not have any issues with 1.58, but if that happens to you, you can install use boost 1.55 as follows (code not tested!):

```
brew uninstall boost boost-python
cd /usr/local
git checkout a252214 /usr/local/Library/Formula/boost.rb
brew install --build-from-source --with-python --fresh -vd boost
brew pin boost  #this will prevent you from inadvertenly upgrading your boost
```


#### Compile Caffe
Lastly, check that Caffe and dependencies are linking against the same, desired Python.  If you used Homebrew to install Python and followed the rest of the instructions above, you should be okay.

Now we will compile Caffe.  First `cd` to the local directory where you want to check out the code and then clone the Caffe repository.

```bash
git clone https://github.com/BVLC/caffe.git
cd caffe
cp Makefile.config.example Makefile.config
```

Since I am using a laptop with no CUDA support, we will modify the `Makefile.config` to use CPU only.  Uncomment the following line:

```
# CPU_ONLY := 1
```

I am also using the Homebrew version of Python, so I will have to update the location information for the Python-related variables.  The default Python library location in `Makefile.config` is: `PYTHON_LIB := /usr/lib`

👉 Note: You may be able to figure out the correct location for your Python installation on your machine by running `locate libpython*`.  This did not return anything on my machine, but it may work for others. 

On my system the relevant .dylib file was located here: 

```bash
/usr/local/Cellar/python/2.7.10/Frameworks/Python.framework/Versions/Current/lib/libpython2.7.dylib
```
👉 Note: If you are using the latest version of Python 2.7 (2.7.11), then just replace the 2.7.10 folder with 2.7.11 in the line above.  

To make sure I use the correct Python, I added the following lines to my `Makefile.config` and commented out the existing `PYTHON_INCLUDE` and `PYTHON_LIB`.  Edit the `BREWPY_HOME` location based on the location of your Python library, found above.

```
BREWPY_HOME := /usr/local/Cellar/python/2.7.10/Frameworks/Python.framework/Versions/Current
PYTHON_INCLUDE := $(BREWPY_HOME)/include \
		 $(BREWPY_HOME)/include/python2.7 \
		 $(BREWPY_HOME)/lib/python2.7/site-packages/numpy/core/include \
```

```bash
PYTHON_LIB := /usr/local/Cellar/python/2.7.10/Frameworks/Python.framework/Versions/Current/lib
```
👉 Note: Make sure you add the updated `PYTHON_INCLUDE` and `PYTHON_LIB` above the following lines in `Makefile.config`, since they use those variables:

```bash
# Whatever else you find you need goes here.
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib
```

Lastly, let's build Caffe!


```
make
make test
make runtest
```
If the tests go well, you will see output that ends with the following lines:

```
[----------] 12 tests from NesterovSolverTest/1 (173 ms total)

[----------] Global test environment tear-down
[==========] 846 tests from 129 test cases ran. (22218 ms total)
[  PASSED  ] 846 tests.
```

#### Caffe for Python
Finally, let's build the Python API for Caffe.  In the main Caffe repository directory, type:

```bash
make pycaffe
```

To import the Caffe Python module after completing the installation, open up your `~/.bashrc` file and add the following line to add the module directory to your `PYTHONPATH`, where `/path/to/caffe` is the location on your machine where you cloned the Caffe git repo:

```bash
export PYTHONPATH=/path/to/caffe/python:$PYTHONPATH 
```

For example, in my case, I added the following line:

```bash
export PYTHONPATH=/Users/me/code/github-repos/caffe/python:$PYTHONPATH
```

👉 Note: If you open a new shell, type `echo $PYTHONPATH`, and it's blank, then you might need to make some [adjustments](http://stackoverflow.com/questions/415403/whats-the-difference-between-bashrc-bash-profile-and-environment) to your `~/.profile` or `~/.bash_profile` file.  For example, adding the following line: `[[ -r ~/.bashrc ]] && . ~/.bashrc` 

Lastly, you will need a few more Python libraries before pycaffe will actually run.  To install the remaining dependencies:

```bash
cd python
for req in $(cat requirements.txt); do pip install $req; done
```


## DeepDream Notebook

### GoogLeNet Model

The original IPython/Jupyter [DeepDream notebook](https://github.com/google/deepdream) makes use of a pre-trained Caffe model which was trained using [GoogLeNet](http://arxiv.org/abs/1409.4842), a 22-layer deep [convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network) (CNN).  More info [here](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet).  It can take a large amount of computing power to train a CNN (aka "CovNet") model, however once the model is trained (probably on a big GPU-based cluster), it can be used easily on your laptop to generate DeepDream images.

You should download the model into the `./models/bvlc_googlenet` folder located in the main Caffe repo. On my machine, I typed:

```bash
cd /Users/me/code/github-repos/caffe/models/bvlc_googlenet
wget http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel
```
👉 Note: If you don't have `wget` installed, you can install it via Homebrew using the following command:  `brew install wget`  If you don't want to install `wget`, then just visit the [model URL](http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel) in your browser instead.


### Caffe Notebook

Now you have all the neccessary tools required to execute the DeepDream IPython/Jupyter notebook.  First `cd` out of the Caffe repo to somewhere else on your computer.  For example:

```bash
cd /Users/me/code/github-repos
```
Then you can clone the DeepDream repo, which contains the `dream.ipynb` file:

```bash
git clone https://github.com/google/deepdream.git
```

Next, we will begin executing the DeepDream notebook.  First, start the notebook server:

```bash
cd deepdream
jupyter notebook  #or ipython notebook
```
This will open a window in your browser, which you can use to click on the `dream.ipynb` file.  This will bring up the notebook.  To execute the cells, simply click on a cell and click "Shift + Enter" to run the code in that cell.

👉 Note: If you see the following error:`ImportError: No module named caffe`, it means that you probably did not set your `PYTHONPATH` properly.  To fix this issue, kill your notebook/Python instance and then make sure your `PYTHONPATH` is set (at least in the terminal from which you launch Python).  If `PYTHONPATH` is blank, Python will not be able to locate the caffe Python module.

In the second block of Python code, there is one line you will need to change, and that's the location of the model file on your local machine.  Look for this line and make the edit accordingly:

```python
model_path = '../caffe/models/bvlc_googlenet/' # substitute your path here
```

For example, I changed mine to:

```python
model_path = '/Users/me/code/github-repos/caffe/models/bvlc_googlenet/'
```

