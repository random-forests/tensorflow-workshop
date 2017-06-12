# Pre-work for the TensorFlow Tutorial

## 1. Install Docker on your laptop

Visit https://docker.com to download and install Docker on your laptop. If you'd like to learn more about Docker, see Docker's getting started guide.

## 2. Once you have installed Docker, make a shared directory in your homedir and get the workshop code:

    $ mkdir tf_workshop
    $ cd tf_workshop
    $ git clone https://github.com/random-forests/tensorflow-workshop.git

## 3. Start the container with a shared directory:

    $ docker run -it -p 8888:8888 -v ~/tf_workshop/tensorflow-workshop/zurich:/notebooks tensorflow/tensorflow 

Beware that your local path (~/tf_workshop/zurich) should start with ~ or /
Using for example "-v ./tf_workshop:/notebooks" won't work.
 
Open the link that looks like ht&#8203;tp://localhost:8888/?token=TOKEN in your browser

## 4. Complete the pre-work notebook

From the main page in your browser, open the **00_download_data.ipynb** from the **Cell** menu, select **Run all** to pre-download the files.

Once you have run all the cells in the download data notebook, you're good to go for the workshop!
