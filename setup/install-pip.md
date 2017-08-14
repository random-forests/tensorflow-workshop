### To install TensorFlow v1.3rc2 using pip

First, follow the [installation instructions](https://www.tensorflow.org/install/) on TensorFlow.org to create a Virtual Environment for your platform, but **stop** before installing TensorFlow with pip (e.g. *do not* run this  command ```pip install --upgrade tensorflow ```)

After activating your virtual environment, run these commands inside it:

```
$ git clone https://github.com/random-forests/tensorflow-workshop.git
$ cd tensorflow-workshop
$ pip install -r setup/requirements.txt
```

This should install the correct TensorFlow version and other libraries within your virtual environment. Now, start a Jupyter Notebook server by running:

```$ jupyter notebook```

If you experience difficulty, try the Docker option on the previous page for an easier option (rather than spending cycles configuring your system to run the RC).