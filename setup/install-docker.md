# Run the TensorFlow Docker image on your laptop

We will present the tutorial in [Jupyter](jupyter.org) notebooks. To
run them on your laptop machine, you will need a working TensorFlow
installation (v1.0+).

Below are instructions on how to set up a TensorFlow environment using
Docker. 

## Step 1: Install Docker on your laptop
Visit [docker.com](https://www.docker.com) to download and install Docker on your laptop. Docker is available for Linux, Mac, and Windows.

Note: the following instructions are written assuming you're using either Linux or Mac. They have not been tested for Windows. If you're using Windows, you may have to modify the syntax slightly for your terminal. For more details on the following commands, see Docker's [getting started guide](https://docs.docker.com/learn/).

If you're stuck, see the Cloud-based [installation](install-cloud.md) guide for an alternative.

## Step 2: Download the container image
Once Docker is installed and running on your laptop: open a terminal, and download the workshop image using this command.

```
$ docker pull tensorflow/tensorflow
```

Note: this is a large download. Now might be a good time to get a cup of coffee. 

## Step 3: Create a shared directory

During the workshop, you'll want to copy data between your laptop and the running container. But, by default, the container's filesystem is isolated from your machine.

So you can share data, create a directory on your laptop, and mount it as a volume when you start the container. Any data inside this folder will be accessible to both your laptop and the container. First, let's create the shared directory:

```
$ mkdir ~/workshop
```

Above, we're creating a directory called ```workshop``` inside your home folder. Feel free to use a different path.

## Step 4. Clone this repository in the shared directory.

Inside the shared directory you just created, use git to clone this workshop.

```
cd ~/workshop
git clone https://github.com/random-forests/tensorflow-workshop.git
cd tensorflow-workshop
```

## Step 5. Start the container

In this step, we'll start the container. We'll use command line arguments to mount your shared directory, and to open two ports we'll need later. Run this command:

```
$ docker run -it -p 8888:8888 -p 6006:6006 -v ~/workshop:/workshop tensorflow/tensorflow bash
```

Long command, right? Here's an explanation of the arguments.

* ```-v ~/workshop:/workshop``` mounts ```~/workshop``` on your laptop as ```/workshop``` inside the container.

* ```-p 8888:8888``` forwards port 8888 on your laptop to port 8888 on the container (this is so we can connect to a Jupyter notebook server running inside the container, more on that soon).

* ```-p 6006:6006``` likewise forwards port 6006. We'll use this later to connect to TensorBoard.

* ```-it``` attaches your terminal to the container.

* ```bash``` starts bash.

Note: at this point, your terminal prompt will change to resemble something like this. All further commands will be run from inside this window.

```
root@83bf8a86752e:/notebooks# 
```

That means your terminal is connected to the running container. Any commands you run in this window will be run inside the container. Try ```ls``` for example.

If you'd like to exit the container and return to your usual terminal, type ```exit```. (You can start the container again with the above command). 

## Step 6. Install dependencies

From inside the container, change into the shared folder and install dependencies by running these commands.
```
cd /workshop/tensorflow-workshop
pip install -r setup/requirements.txt
```

## Step 7. Start a Jupyter Notebook server

In this step, we will start an Jupyter Notebook server that runs inside the container. We will then connect to it using a web browser on your laptop.

**Step 7a)** Start the notebook server.

From inside the container:

```
jupyter notebook --allow-root
```

You will see output on your terminal to indicate the server is running. Later, if you want to stop the notebook server, press *Control-C*. 

**Step 7b)** Copy the login token

Notice the second from the last line contains a login token. Copy this; you will need it in a moment to connect to the server.

**Step 7c)** Use a web browser to connect to the server

Open a web browser on your laptop. Enter ```localhost:8888``` in the address bar to connect to the server. Paste the login token you coped in the previous step when prompted.

## When you're done

To exit Docker, you can simply enter `exit` or hit `Ctrl-D`. 

Note: by default, any data you create inside the Docker container will be lost when it exits. To save data, copy it to the shared directory before stopping the container.
