# Welcome

# Installation instructions

For this workshop, you may either install TensorFlow directly on your laptop, or use a Docker container running either locally, or on a Cloud-based VM. Please choose the option below that's right for you.

### Option 1: Install TensorFlow directly on your laptop
This approach is recommended if you prefer the smallest download.

1. [Install TensorFlow on Mac using a Virtual Environment.](setup/install-mac.md) 

1. [Install TensorFlow on Windows.](setup/install-win.md)

1. [Install TensorFlow on Ubuntu/Linux using a Virtual Environment.](setup/install-ubuntu.md)

If you're an experienced developer, you can find a pip requirements file [here](setup/requirements.txt).

### Option 2: Install TensorFlow using Docker

This approach is a good option if you have difficulty installing TensorFlow directly.

1. [Run the TensorFlow Docker image on your laptop.](setup/install-docker.md)

1. [Run the TensorFlow Docker image on a Cloud-based VM.](setup/install-cloud.md) This is a good option if you have difficulty using Docker on your laptop.

---
If you get stuck, please refer to the full installation [instructions](https://www.tensorflow.org/install/) on tensorflow.org.

# Install other Dependencies

* **For Linux only**, run: `$ sudo install_dependencies.sh --python_version (2 or 3)`, python_version is an optional argument the script will install the dependencies for both versions of Python 2 and 3 if not explictly defined. For more details check [install_dependices.sh](install_dependencies.sh).

  Or Install dependencies manyally:
    * `python2 -m pip install jupyter numpy pandas matplotlib pillow`
    * `python3 -m pip install jupyter numpy pandas matplotlib pillow`

For other SOs check the next section.

**If you are not familiar with [Jupyter notebooks](http://jupyter.readthedocs.io/en/latest/index.html) please visit the website and learn more about it.**
   
## Test if everything is working properly

1. Clone this repo or download it
2. cd tensorflow-workshop
3. jupyter notebook
4. run test_install.ipynb
5. If you see any errors with the libraries, make sure you have them installed and run test_install.ipynb again.

# Want to report a bug?

Thanks! Can you please file an issue, or even better, a pull request? We'll be doing this workshop a couple times, and future developers will appreciate your help.

- - -

General disclaimer, this is my personal repo and not an official Google product. If you'd like to use this code, say, to build a mission critical component of your giant space laser, you should know there's no warranty, etc.
