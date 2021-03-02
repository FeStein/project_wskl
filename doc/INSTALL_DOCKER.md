# Install Docker Tutorial
This file covers the installation of docker on a Linux machine (Ubuntu) in a
short manner in order to work with the python tools for my project. For a more
comprehensive installation guide please refer the [official docker documentation](https://docs.docker.com/engine/install/ubuntu/).
If you can run docker properly on any other device the relevant containers
should work as well.

## 1. Install Graphic Drivers

In order to enable GPU support (my images are built on CUDA, so GPU support is
recommended) install the proprietary nvidia drivers. The steps are as follows:

* Check that you have a nvidia gpu installed via  `sudo lshw -C display`
* Open the sotware update manager
* Click on *settings* 
* Click on *additional drivers tab*
* chose the desired nvidia driver (you mostly want **nvidia-driver-460
    (proprietary, tested)**)
* Apply changes 
* Reboot

A visual guide can be found
[here](https://www.cyberciti.biz/faq/ubuntu-linux-install-nvidia-driver-latest-proprietary-driver/)

## 2. Install Docker

The following steps will set up the docker repository in your package manager
(assuming you use apt) and install the software.

### 2.1 Remove prior versions (if necessary)

```bash
sudo apt-get remove docker docker-engine docker.io containerd runc
```

### 2.2 Update apt and install necessary requirements (if not already done)

```bash
sudo apt-get update
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
```

### 2.3 Add GPG Key

```bash
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
```
If you want to verify see [here](https://docs.docker.com/engine/install/ubuntu/)

### 2.4 Set up repo and install (assuming x86_64 architecture)
```bash
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
```

### 2.5 Verify installation
You may need sudo rights:
```bash
sudo docker run hello-world
```
It should download the image (can take some time) and the result should look
like this:
```bash
Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
1. The Docker client contacted the Docker daemon.
2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
(amd64)
3. The Docker daemon created a new container from that image which runs
the
executable that produces the output you are currently reading.
4. The Docker daemon streamed that output to the Docker client,
which sent it
to your terminal.

To try something more ambitious, you can run an Ubuntu
container with:
$ docker run -it ubuntu bash

Share images, automate workflows, and more with a free
Docker ID:
https://hub.docker.com/

For more examples and ideas, visit:
https://docs.docker.com/get-started/

```
If everything works correctly you're good to go and you can run any image you
want.

## Post installation steps (optional)

If you want convenient settings for your docker installation (for instance not
using budo every time you try to run a docker image) see
[here](https://docs.docker.com/engine/install/linux-postinstall/).






