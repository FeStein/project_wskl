# Installing Docker on Linux Mint with Nvidia GPU Support
-> generally should not work out of the box, since the nvidia container toolkit
does not officially support linux mint.

## Relevant Links
* [installation for Linux Mint (based on ubuntu1604)](https://marmelab.com/blog/2018/03/21/using-nvidia-gpu-within-docker-container.html)
* [nvidia github page](https://nvidia.github.io/nvidia-docker/)
* [Nvidia install page](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

## Possible Solution

* The Ubuntu 20.04 Version is the Ubuntu18.04 version of the nvidia container
        toolkit (see associated file in the *nvidia-docker.list* file copyed
        from my working ubuntu 20.04 machine).

Use the following commands:
```bash
distribution=$(ubuntu20.04) \
  && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-keyadd - \
  && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```
Then try the working example
```bash
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```







