# Run the TensorFlow Docker image on a Cloud-based VM
These instructions will show you how to start a Google Compute Engine (GCE) VM with Docker installed, and how to run a TensorFlow container there. This is a good option if you have difficulty installing TensorFlow on your laptop, or if you prefer to work in the Cloud. 

<a name="create"></a>
## Initial setup

### 1. Create a Google Cloud Platform account

Sign up for a [free trial](https://cloud.google.com/free-trial/) of Google Cloud Platform (GCP). You will need a credit card to sign up, and you will receive $300 of free credits. Note: at the time of writing, you will not be billed unless you decide to renew after the trial ends. 

**Note:** The author of this workshop is not interested in selling you Cloud services. GCP is pretty cool, but if you prefer, of course feel free to to use an alternative provider. The following instructions are written assuming you are using Google Cloud.

#### 1.1 Enable the necessary APIs

1. Go to the Google Cloud console: [console.cloud.google.com](https://console.cloud.google.com)
1. Select or create a project using the project drop-down at upper-left ('My First Project' in image below)
1. Click on the “hamburger” menu at upper-left, and then “API Manager”.
1. On the left nav, choose "Dashboard" if not already selected, then choose "+Enable API" in the top-middle of page.
1. Enter "Google Compute Engine API" in the search box and click it when it appears in the list of results.
1. Click on “Enable” (top-middle of page).

![Hamburger menu](../images/hamburger.png)  

![API Manager](../images/api_manager.png)

### 2. Connect to your project's Cloud Shell

Click on the Cloud Shell icon (leftmost icon in the set of icons at top-right of the page).

![Cloud Shell](../images/cloudshell2.jpg)

Click on "Start Cloud Shell" on the bottom right of the pop-up screen. You should now see a terminal at the bottom of your window for the Cloud Shell with the text "Welcome to Cloud Shell! Type "help" to get started."

### 3. Create a container-optimized image in GCE

Run this command in the Cloud Shell.

```shell
gcloud compute instances create workshop \
    --image-family gci-stable \
    --image-project google-containers \
    --zone us-central1-b --boot-disk-size=100GB \
    --machine-type n1-standard-1
```

After you run this command, you can ignore the "I/O performance warning for disks < 200GB".

### 4. Set up a firewall rule for your project that will allow access to the IPython notebook server and TensorBoard

```shell
gcloud compute firewall-rules create workshop --allow tcp:8888,tcp:6006
```

### 5. SSH into the new GCE instance, in a new browser window

- Click on the “hamburger” menu at upper-left, and then “Compute Engine”
- Find your instance in the list (mid-page)
- Write down the "External IP", this is the IP of your Cloud instance
- Logon to instance by clicking on the “SSH” pulldown menu on the right. Select “Open in browser window”.
- A new browser window will open, with a command line into your GCE instance. Confirm that you wish to initiate an SSH connection to the instance.

### 6. Start the Docker container in the GCE image

In the SSH browser window that's connected to the GCE instance, run this command to download and run the container:

```shell
$ docker run -it -p 8888:8888 p 6006:6006 tensorflow/tensorflow bash
```

When this command completes, the terminal in your SSH browser window will be connected to the running container.

### 7. Clone the workshop

Clone this workshop inside the container. In the SSH browser window that's connected to the container, run:

```
# git clone https://github.com/random-forests/tensorflow-workshop.git
```

### 8. Start a notebook server

In this step, we will start an IPython Notebook server that runs inside the container. We will then connect to it using a web browser on your laptop (or Chromebook). In the SSH browser window that's connected to the container, run:

```
# cd tensorflow-workshop
# jupyter notebook
```

You will see output on your terminal to indicate the server is running. If you want to stop the notebook server later, press *Control-C* (but do not do this now).

**Step 9a)** Copy the login token.

When the Jupyter server starts, you'll see a lot of console output. One one of the last lines, you'll see a login token. Copy this; you will need it in a moment to connect to the server.

**Step 9b)** Use a web browser on your laptop to connect to the notebook server

Open a web browser on your laptop. Use the "External IP" of your your Cloud VM from step #5 followed by :8888, i.e. ```<External_IP>:8888``` in the address bar.

Paste the login token you coped in the previous step to connect to the server (if you copy/paste, make sure there is no newline splitting the token value, i.e. that you are pasting the token value printed on a single line). 

**Step 9c)** Test your install

Open the ```examples``` folder, and click on *00_test_install.ipynb*. You should be able to run the notebook without issue.

That's it! You're ready to begin the workshop. Leave the container and Jupyter Notebook server running.

## Cleanup

Once you’re done with the workshop, you can stop or delete your GCE instance. If you think you might return to it later, you might prefer to just stop it. (A stopped instance does not incur charges, but all of the resources that are attached to the instance will still be charged).  You can do this from the [cloud console](https://console.cloud.google.com), or via command line from the Cloud Shell as follows:

```shell
gcloud compute instances delete --zone us-central1-b workshop
```
Or:

```shell
gcloud compute instances stop --zone us-central1-b workshop
```
Then later:

```shell
gcloud compute instances start --zone us-central1-b workshop
```
Delete the firewall rule as well:

```shell
gcloud compute firewall-rules delete workshop
```