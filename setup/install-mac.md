# Install TensorFlow on Mac using a Virtual Environment

We will present the tutorial in [Jupyter](http://jupyter.org) notebooks. To run them on your laptop, you will need a working TensorFlow installation (v1.0+) and several other libraries. 

Follow these instructions, which assume you have a recent version of OSX (probably 10.12), and will use Python 2.7.

## Open a terminal.

Open `Terminal`. This tutorial assumes you are using `bash`, which you
probably are.

## Clone this repository

Using git, clone this tutorial and enter that directory.

```
git clone https://github.com/random-forests/tensorflow-workshop.git
cd tensorflow-workshop
```

## Install Pip and Virtualenv

Pip is a package management system used to install and manage software
packages written in Python.  Virtualenv allows you to manage multiple
package installations.

At your Terminal window, run the following command. 
```
# Mac OS X
sudo easy_install --upgrade pip
```

Once you've installed pip, you'll need to add a few more packages.

```
sudo easy_install --upgrade six
sudo pip install --upgrade virtualenv
```

Now, create a virtual environment.

```
virtualenv --system-site-packages ~/tensorflow
```

> Note: If you have already installed anaconda, some versions of
> anaconda and virtualenv are not compatible.  If you have trouble,
> such as seeing errors about "sys.prefix", you may want to try to
> use the [TensorFlow anaconda installation instructions](https://www.tensorflow.org/install/install_mac#installing_with_anaconda).

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

From your "tensorflow" virtualenv prompt, run the following command to start a Jupyter Notebook server:

```
(tensorflow) $ jupyter notebook
```

Click on `00_test_install.ipynb` to test that jupyter is running
correctly.

You should be able to run the code in notebook without any import errors.

## Installation notes

Virtualenv is a tidy way of managing your dependencies.  Any time
you want to run TensorFlow, you can activate the virtual environment by `source
~/tensorflow/bin/activate`.  To exit the virtual environment, simply
type `deactivate`.

Without using Virtualenv, at this time you may run into issues with
upgrading some pre-installed Python dependencies on MacOS.