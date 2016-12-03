# Installing TensorFlow on Ubuntu

We will present the tutorial in [Jupyter](jupyter.org) notebooks.  To
run them on your machine, you will need a working TensorFlow
installation (v0.11+).

Follow these instructions, which assume you have Ubuntu Linux.  Other
Linux distributions may not be supported.


## Clone this repository.

```
git clone https://github.com/random-forests/tensorflow-workshop.git
cd tensorflow-workshop
```

## Install pip and dependencies

```
sudo apt-get install python-pip python-dev python-matplotlib
```

## Install TensorFlow

This uses CPU only, and requires `pip` to be at version 8.1 or greater.

```
sudo pip install --upgrade tensorflow
sudo pip install jupyter Pillow
```

For GPU instructions or alternate installtion instructions (such as
using Anaconda, or if you have an old version of pip you cannot
upgrade), see
[tensorflow.org](https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html).

## Running Jupyter

Run the following:

```
jupyter notebook
```

Click on `0_test_install.ipynb` to test that jupyter is running
correctly.

You should be able to run the notebook without issue.

<hr>
