# Pre-work for the TensorFlow Tutorial

1. Install Docker on your laptop

Visit docker.com to download and install Docker on your laptop. If you'd like to learn more about Docker, see Docker's getting started guide.

2. Download the container image

Once you have installed Docker: open a terminal and download the workshop image using this command.

3. Make a shared directory in your homedir and get the workshop code:

    mkdir tf_workshop
    cd tf_workshop
    git clone https://github.com/random-forests/tensorflow-workshop.git

4. Start the container with a shared directory:

    docker run -it -p 8888:8888 -v ~/tf_workshop/tensorflow-workshop/zurich:/notebooks tensorflow/tensorflow 

Beware that your local path (~/tf_workshop/zurich) should start with ~ or /
Using for example "-v ./tf_workshop:/notebooks" won't work.
 
Open the link that looks like http://localhost:8888/?token=<TOKEN> in your browser
 
4. Complete the pre-work notebook
 
From the main page in your browser, open the 0_DownloadData.ipynb and pre-download the files.

Pre-work for the TensorFlow Tutorial
 
1. Install Docker on your laptop
 
Visit docker.com to download and install Docker on your laptop. If you'd like to learn more about Docker, see Docker's getting started guide.
 
2. Download the container image
 
Once you have installed Docker: open a terminal and download the workshop image using this command.
 
3. Make a shared directory in your homedir and get the workshop code:
 
mkdir tf_workshop
cd tf_workshop
git clone https://github.com/random-forests/tensorflow-workshop.git
 
4. Start the container with a shared directory:
 
docker run -it -p 8888:8888 -v ~/tf_workshop/tensorflow-workshop/zurich:/notebooks tensorflow/tensorflow 


Beware that your local path (~/tf_workshop/zurich) should start with ~ or /
Using for example "-v ./tf_workshop:/notebooks" won't work.
 
Open the link that looks like http://localhost:8888/?token=<TOKEN> in your browser
 
4. Complete the pre-work notebook

From the main page in your browser, open the 00_DownloadData.ipynb from the **Cell** menu, select **Run all** to pre-download the files.

Once you can load the notebooks and have run all the cells in the download data notebook, you're good to go for the workshop!
