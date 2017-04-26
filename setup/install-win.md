# Install TensorFlow on Windows

We will present the tutorial in [Jupyter](jupyter.org) notebooks.  To
run them on your Windows machine, you will need a working TensorFlow
installation (v1.0+).

Follow these instructions, which assume you have Windows 7 or 10.
Other versions of Windows may not be compatible.

## Be sure you are ready

To run TensorFlow on Windows, you will need to install Python 3.5 and
git.  You can find installation instructions on
[python.org](python.org).  Note that you will need a *64-bit*
installation.

## Clone this repository.

Open a shell (`cmd.exe` or whatever kind of shell you use).

```
git clone https://github.com/random-forests/tensorflow-workshop.git
cd tensorflow-workshop
```

## Install TensorFlow and dependencies

If you install Python from python.org, you already have `pip`, so you
can proceed directly to installing TensorFlow and dependencies.

```
# Run this command from inside the directory 
# where you cloned this workshop
pip install -r setup/requirements.txt
```

## Running Jupyter

From the command prompt, run the following from the `tensorflow-workshop` directory:

```
jupyter notebook
```

Open the ```examples``` folder, and click on *00_test_install.ipynb*. You should be able to run the notebook without issue.

## Possible issues

TensorFlow requires `MSVCP140.DLL`, which may not be installed on your
system.  If, when you `import tensorflow as tf`, you see an error
about `No module named "_pywrap_tensorflow"` and/or `DLL load failed`,
check whether `MSVCP140.DLL` is in your `%PATH%` and, if not, you
should install the [Visual C++ 2015
redistributable](https://www.microsoft.com/en-us/download/details.aspx?id=53587)
(x64 version).

For GPU or alternate installation instructions, see [tensorflow.org](https://www.tensorflow.org/install/).