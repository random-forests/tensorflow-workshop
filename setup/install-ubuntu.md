# Install TensorFlow on Ubuntu using a Virtual Environment

We will present the tutorial in [Jupyter](jupyter.org) notebooks.  To
run them on your machine, you will need a working TensorFlow
installation (v1.0+).

Follow these instructions, which assume you have Ubuntu Linux.  Other
Linux distributions may not be supported.


## Clone this repository.

```
git clone https://github.com/random-forests/tensorflow-workshop.git
cd tensorflow-workshop
```

## Install pip and virtualenv

```
sudo apt-get install python-pip python-dev python-virtualenv python-tk
```

Now, create a virtual environment.

```
virtualenv --system-site-packages ~/tensorflow
```

You will need to *activate* the environment, which is to say, switch your
Python enviroment to a fresh one with clean dependencies.

```
source ~/tensorflow/bin/activate
```

You are now running in a special Python enviroment with safe
dependencies. Your prompt should start with `(tensorflow) $`.

Run these commands to install TensorFlow, Jupyter, and other software.

```
# Within the (tensorflow) virtualenv,
# run this command from inside the directory 
# where you cloned this workshop
pip install -r setup/requirements.txt
```

## Running Jupyter

Run the following:

```
jupyter notebook
```

Open the ```examples``` folder, and click on *00_test_install.ipynb*. You should be able to run the notebook without issue.

## Installation notes

Virtualenv is a tidy way of managing your dependencies.  Any time
you want to run TensorFlow, you can activate the virtual environment by `source
~/tensorflow/bin/activate`.  To exit the virtual environment, simply
type `deactivate`.

For GPU or alternate installation instructions, see [tensorflow.org](https://www.tensorflow.org/install/).