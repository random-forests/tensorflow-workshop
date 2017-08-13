# How-to Install TensorFlow using Docker on your laptop

*These instructions will guide you through downloading and running the TensorFlow Docker container. They'll also show you how to start a [Jupyter](jupyter.org) notebook server, and connect to it from your browser. There's a lot of text ahead, but only a few commands to actually run.* 

We will present the tutorial in [Jupyter](jupyter.org) notebooks. To
run them on your laptop machine, you will need a working TensorFlow
installation (v1.3+).

## Step 1: Install Docker on your laptop
Visit [www.docker.com](https://www.docker.com) to download and install Docker on your laptop. Docker is available for Linux, Mac, and Windows. To learn more about Docker in general, you can see the [getting started guide](https://docs.docker.com/learn/) - or you can proceed fearlessly ahead.

## Step 2: Download the container image
Once Docker is installed and running on your laptop: open a terminal, and download the TensorFlow container using this command.

```
$ docker pull tensorflow/tensorflow:1.3.0-rc2
```

Note: this is about 380 MB. 

## Step 3: Create a shared directory

During the workshop, you may wish to copy data between your laptop and the running container. By default, the container's filesystem is isolated from your machine. Here, we will create a directory on your laptop, and mount it as a volume when you start the container. Any data inside this folder will be accessible to both your laptop and the container. 

First, let's create the directory we'll mount in your home folder. Feel free to use a different path.

```
$ mkdir ~/shared
```

## Step 4. Start the container

Run this monster command to start the container.

```
$ docker run -it -p 8888:8888 -p 6006:6006 -v ~/shared:/shared tensorflow/tensorflow:1.3.0-rc2 bash
```

Long command, right? Here's what the arguments do:

* ```-v ~/shared:/shared``` mounts ```~/shared``` on your laptop as ```/shared``` inside the container.


* ```-p 8888:8888``` forwards port 8888 on your laptop to port 8888 on the container. We'll use this to connect to the Jupyter Notebook server.


* ```-p 6006:6006``` forwards port 6006. We'll use this to connect to TensorBoard.


* ```-it``` attaches your terminal to the container.


* ```bash``` starts bash.

At this point, the container will start - and your terminal prompt will change to resemble something like this. 

```
root@83bf8a86752e:/notebooks# 
```

This means your terminal is connected to the running container. Any commands you run in this window will be run inside the container. Try ```ls``` for example, and you will see the containers filesystem.

If you'd like to exit the container and return to your usual terminal, type ```exit```. (You can start the container again with the above command). 

## Step 5. Clone the workshop

*All further commands will be run from inside the terminal connected to the running container.*

We will now clone today's workshop inside your shared directory.

```
# cd /shared
# git clone https://github.com/random-forests/tensorflow-workshop.git
# cd tensorflow-workshop
```

## Step 6. PIP install a few libraries

We will now use pip to install a few libraries. Because we're running inside the docker container, these will not affect your system at all.

```
# pip install -r setup/requirements.txt
```

## Step 7. Start a Jupyter Notebook server

We will now start a Jupyter Notebook server running inside the container. We will then connect to it using a web browser on your laptop.

**Step 7a)** Start the notebook server.

From inside the container:

```
# jupyter notebook --allow-root
```

You will see output on your terminal to indicate the server is running. Later, if you want to stop the notebook server, press *Control-C*. 

**Step 7b)** Copy the login token

Notice the second from the last line contains a login token. Copy this; you will need it in a moment to connect to the server.

**Step 7c)** Use a web browser to connect to the server

Open a web browser on your laptop. Enter ```localhost:8888``` in the address bar to connect to the server. Paste the login token you coped in the previous step when prompted.

Open the ```examples``` folder, and click on *00_test_install.ipynb*. You should be able to run the notebook without issue.

## When you're done

Stop the Jupyter Server by pressing `Ctrl-C` then `y`.

To exit Docker, type `exit` from the terminal.

Note: by default, any data you create inside the Docker container will be lost when it exits. To save data, copy it to the shared directory before stopping the container.