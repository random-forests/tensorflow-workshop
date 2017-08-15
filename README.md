# Welcome

##  Install TensorFlow version 1.3rc2

This code requires **TensorFlow version 1.3** which at the time of writing, has not been released. You can install the release candidate (v1.3rc2) using these instructions. Choose the option that's right for you:

### 1. Install the RC using Docker

This approach is the easiest option. It works similarly on Mac, Linux, and Windows, and is recommended if you have difficulty installing the RC in a virtual environment. 

* [How-to run the TensorFlow Docker image on your laptop.](setup/install-docker-local.md)

* [How-to run the TensorFlow Docker image on GCP.](setup/install-docker-cloud.md)

### 2. Install the RC using pip in a Virtual Environment

This approach is recommended if you prefer the smallest download. If you experience difficulty, try the Docker option above.

* [How-to install the RC using pip.](setup/install-pip.md)

---
If you get stuck, please refer to the full installation [instructions](https://www.tensorflow.org/install/) on tensorflow.org.

# Install other Dependencies

* **For Linux only**, run: `$ sudo install_dependencies.sh --python_version (2 or 3)`, python_version is an optional argument the script will install the dependencies for both versions of Python 2 and 3 if not explictly defined. For more details check [install_dependices.sh](install_dependencies.sh).

  Or Install dependencies manyally:
    * `python2 -m pip install jupyter numpy pandas matplotlib pillow`
    * `python3 -m pip install jupyter numpy pandas matplotlib pillow`

For other OS check the next section.

**If you are not familiar with [Jupyter notebooks](http://jupyter.readthedocs.io/en/latest/index.html) is recommened to visit the website and learn more about it.**
   
## Test if everything is working properly

1. Clone this repo or download it
2. $ cd tensorflow-workshop
3. $ jupyter notebook
4. run test_install.ipynb
5. If you see any errors with the libraries, make sure you have them installed and run test_install.ipynb again.

# Want to report a bug?

Thanks! Can you please file an issue, or even better, a pull request? We'll be doing this workshop a couple times, and future developers will appreciate your help.

- - -
General disclaimer, this is my personal repo and not an official Google product. If you'd like to use this code, say, to build a mission critical component of your giant space laser, you should know there's no warranty, etc.
