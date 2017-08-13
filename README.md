# Welcome

##  Install TensorFlow version 1.3rc2

Welcome! This code requires **TensorFlow version 1.3** which at the time of writing, has not been released. You can install the release candidate (v1.3rc2) using these instructions.

### 1. Install the RC using Docker

This approach is the easiest option, works similarly on Mac, Linux, and Windows, and is recommended if you have difficulty installing the RC in a virtual environment.

* [Run the TensorFlow Docker image on your laptop.](setup/install-docker-local.md)

* [Run the TensorFlow Docker image on GCP.](setup/install-docker-cloud.md)

### 2. Install the RC using pip in a Virtual Environment

This approach is recommended if you prefer the smallest download.

Follow the [installation instructions](https://www.tensorflow.org/install/) on TensorFlow.org to create a Virtual Environment for your platform, but **stop** before installing TensorFlow with pip (e.g. *do not* run this  command ```pip install --upgrade tensorflow ```)

After activating your virtual environment, run these commands inside it:

```
$ git clone https://github.com/random-forests/tensorflow-workshop.git
$ cd tensorflow-workshop
$ pip install -r setup/requirements.txt
```

This should install the TensorFlow RC and other libraries within your virtual environment. Now, start a Jupyter Notebook server by running:

```$ jupyter notebook```

### Want to report a bug?

Thanks! Can you please file an issue, or even better, a pull request? We'll be doing this workshop a couple times, and future developers will appreciate your help.

- - -
General disclaimer, this is my personal repo and not an official Google product. If you'd like to use this code, say, to build a mission critical component of your giant space laser, you should know there's no warranty, etc.