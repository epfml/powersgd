#!/bin/bash
set -e  # stop the script whenever something fails

# Start from a clean Ubuntu 18.04 installation

# Manual steps:
# rsync files/ to hostname:
# Copy jobmonitor*.whl to ~/install

mkdir -p ~/install
cd ~/install

# Install important packages
sudo apt update
sudo apt upgrade -y
sudo apt install -y \
    autoconf \
    autogen \
    build-essential \
    bzip2 \
    ca-certificates \
    cmake \
    curl \
    flex \
    git \
    libevent-dev \
    libffi-dev \
    libglib2.0-0 \
    libjpeg-dev \
    libmunge-dev \
    libmunge2 \
    libnuma-dev \
    libpng-dev \
    libsm6 \
    libssl-dev \
    libtool \
    libxext6 \
    libxrender-dev \
    locales \
    munge \
    nfs-common \
    nfs-kernel-server \
    openssh-server \
    rsync \
    supervisor \
    tmux \
    unzip \
    vim \
    wget \
    zsh


# Install CUDA 11.0
# from https://developer.nvidia.com/cuda-11.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda-repo-ubuntu1804-11-0-local_11.0.2-450.51.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-11-0-local_11.0.2-450.51.05-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu1804-11-0-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda

# Install CUDNN
CUDNN_FILE=libcudnn8_8.0.4.30-1+cuda11.0_amd64.deb
CUDNN_DEV_FILE=libcudnn8-dev_8.0.4.30-1+cuda11.0_amd64.deb
for file in $CUDNN_FILE $CUDNN_DEV_FILE;
do
    wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/$file
    sudo dpkg -i $file
    rm $file
done

CUDART_FILE=cuda-cudart-11-0_11.0.221-1_amd64.deb
for file in $CUDART_FILE;
do
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/$file
    sudo dpkg -i $file
    rm $file
done

# NVML lets Slurm automatically detect GPUs
# LICENSE_FILE=cuda-license-10-1_10.1.243-1_amd64.deb
NVML_FILE=cuda-nvml-dev-11-0_11.0.167-1_amd64.deb
for file in $NVML_FILE;
do
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/$file
    sudo dpkg -i $file
    rm $file
done

# Install  NCCL
NCCL_FILE=libnccl2_2.8.3-1+cuda11.0_amd64.deb
NCCL_DEV_FILE=libnccl-dev_2.8.3-1+cuda11.0_amd64.deb
for file in $NCCL_FILE $NCCL_DEV_FILE;
do
    wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/$file
    sudo dpkg -i $file
    rm $file
done

# Install  NvInfer
sudo apt install cuda-nvrtc-11-0
NVINFER_FILE=libnvinfer7_7.2.2-1+cuda11.0_amd64.deb
NVINFER_PLUGIN_FILE=libnvinfer-plugin7_7.2.2-1+cuda11.0_amd64.deb
for file in $NVINFER_FILE $NVINFER_PLUGIN_FILE;
do
    wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/$file
    sudo dpkg -i $file
    rm $file
done


export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.profile
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.profile


# Create a directory to install cluster software in
export INSTALL_PREFIX=/usr/cluster
sudo mkdir -p $INSTALL_PREFIX
sudo chown -R $USER:$USER $INSTALL_PREFIX


# Install PMIx
wget https://github.com/openpmix/openpmix/releases/download/v3.1.5/pmix-3.1.5.tar.gz
tar xvf pmix-3.1.5.tar.gz
rm pmix-3.1.5.tar.gz
pushd pmix-3.1.5
./configure --prefix=$INSTALL_PREFIX
make -j install
popd


# Install UCX
git clone https://github.com/openucx/ucx.git
pushd ucx
./autogen.sh
./configure --prefix=$INSTALL_PREFIX --with-cuda=/usr/local/cuda
make -j install
popd


# Install OpenMPI
git clone https://github.com/open-mpi/ompi.git
pushd ompi
git checkout v4.0.3
./autogen.pl
./configure --prefix=$INSTALL_PREFIX --with-cuda=/usr/local/cuda --with-ucx=$INSTALL_PREFIX --with-pmix=$INSTALL_PREFIX
make -j install
popd

echo 'export OMPI_MCA_opal_cuda_support=true' >> ~/.profile
export OMPI_MCA_opal_cuda_support=true


# Install and configure slurm
git clone https://github.com/SchedMD/slurm.git
pushd slurm
git checkout slurm-20-02-1-1
./configure --prefix=$INSTALL_PREFIX --sysconfdir=/etc/slurm --with-ucx=$INSTALL_PREFIX --with-pmix=$INSTALL_PREFIX --with-hdf5=no
make -j install
popd
sudo useradd -M slurm
sudo mkdir -p /etc/slurm /etc/slurm/prolog.d /etc/slurm/epilog.d /var/spool/slurm/ctld /var/spool/slurm/d /var/log/slurm
sudo chown -R slurm:slurm /var/spool/slurm /var/log/slurm
sudo chmod -R +w /var/spool/slurm /var/log/slurm
sudo cp ~/slurm/*.conf /etc/slurm
sudo cp ~/slurm/*.sh /etc/slurm
sudo touch /var/log/power_save.log
sudo cp ~/slurm/*.service /etc/systemd/system/
sudo chown -R slurm:slurm /var/spool/slurm /etc/slurm
sudo chown slurm:slurm /var/log/power_save.log
sudo chmod 644 /var/log/power_save.log
sudo systemctl enable slurmctld
sudo systemctl enable slurmd

export PATH=$INSTALL_PREFIX/bin:$INSTALL_PREFIX/sbin${PATH:+:${PATH}}
echo "export PATH=$INSTALL_PREFIX/bin:$INSTALL_PREFIX/sbin"'${PATH:+:${PATH}}' >> ~/.profile
export LD_LIBRARY_PATH=$INSTALL_PREFIX/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
echo "export LD_LIBRARY_PATH=$INSTALL_PREFIX/lib"'${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.profile


# Install Anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
chmod +x Anaconda3-2020.02-Linux-x86_64.sh
sudo ./Anaconda3-2020.02-Linux-x86_64.sh -b -p /opt/anaconda3
rm Anaconda3-2020.02-Linux-x86_64.sh

export PATH=/opt/anaconda3/bin${PATH:+:${PATH}}
echo 'export PATH=/opt/anaconda3/bin${PATH:+:${PATH}}' >> ~/.profile


# Make sure root can find things in our custom paths
echo 'Defaults    secure_path="'$INSTALL_PREFIX'/bin:'$INSTALL_PREFIX'/sbin:/opt/anaconda3/bin:'$HOME'/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin"' | sudo tee /etc/sudoers.d/custom_path


# Build PyTorch (this takes a while -- make sure to check in the beginning that MKL, CUDNN and OpenMPI w/ CUDA support are found)
sudo conda install -y numpy ninja pyyaml mkl mkl-include setuptools cmake cffi
sudo conda install -y -c pytorch magma-cuda110
sudo conda install -y mkl-include  # don't know why this is necessary, but it wasn't there ...
git clone --recursive  https://github.com/pytorch/pytorch
pushd pytorch
git checkout master
git submodule sync
git submodule update --init --recursive
CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"} \
    TORCH_CUDA_ARCH_LIST="3.7 6.0 7.0+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    python setup.py install --user
popd


# Install pip packages
pip install --user --upgrade \
    black \
    fairseq \
    google-cloud-storage \
    influxdb \
    kubernetes \
    msgpack-numpy \
    pyarrow \
    pytelegraf \
    sklearn \
    networkx \
    tensorflow \
    tensorflow_datasets \
    tensorflow_federated \
    spacy \
    lmdb \
    opencv-python

pip install --user --no-deps torchvision torchtext

python -m spacy download en

# Add pip's bin directory to the path
export PATH=$HOME/.local/bin${PATH:+:${PATH}}
echo 'export PATH=$HOME/.local/bin${PATH:+:${PATH}}' >> ~/.profile


# Install bit2byte
git clone https://github.com/tvogels/signSGD-with-Majority-Vote.git
pushd signSGD-with-Majority-Vote/main/bit2byte-extension/
python setup.py install --user
popd


# Install GCFUSE to mount gs buckets
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install gcsfuse -y
sudo rm /usr/local/bin/gcsfuse
sudo ln -s /user/bin/gcsfuse /usr/local/bin/gcsfuse
sudo mkdir -p /gs-bucket
sudo chmod 777 /gs-bucket

# Ensure passwordless access between workers
echo "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQDQVy8uRZwMnJfyGRohzJjOJb3TTdMj0RHyY/+MGt1qElkARyjtdyxFRnZRsEXDjYNnA8fS2m1GXU0pIZIUq+TBs0CAARvz3jRxmo0V7vxl6T9QQsX7bWb4Vg8N48pndFurtVdRzAiE2yfT+jueZkZYBFxWQXZ3zAQu3uXIl+hp/OCJjYedTfrvZfXvUsuU9tHNapWMnh4bwiUDQxKYXzBqdG7SABNTxQ1NJb3+1Jtt1rZ5nUxujnrRWFEhCCyLs4ePISM3U4oqwCjTdFmVq4IZC7bTdcQmB7xDJAX0m8iGG9PtnDnTqOkwZvd4T45x5CCsWuVgghsBYCHGRBFX4kYwLIMG+eoKwtxfJ3qpl0qWSZgweA+go77RM59chIUVoR+E9UYmiYIwINZ699gk9ubOYjNlMDwU0y0ODbPEKHseF/xcILAzpokcD1qJR7rBWlpuqyADJI7R2ZCDaRfBEvCbJZGav2aR5MJBWMt/dy/TPKuzKscpEsIriEB3m7FcA49xvks7fLm3vRDEqafAXuMAEQXGrufnglXb53esdUD6s3QEuX1XLzvVDeD741OAMYRMBarTlCY38RczyaBL1iY50tLs8Hc4f7qVR3ssEZiw0Rk7c3uY8/LMKu2tjcuCgL98twdQgWVndsWpcToADAi0zZ/V1RbcKjsUcIY2jtZveQ== gcloud" >> ~/.ssh/authorized_keys
echo "Host *" >> ~/.ssh/config
echo "    StrictHostKeyChecking no" >> ~/.ssh/config


# Install and configure jobmonitor
# MANUAL: scp jobmonitor.whl to the machine by hand
pip install --user --upgrade jobmonitor*.whl

echo 'export DATA=/mnt/cluster' >> ~/.profile
echo 'export JOBMONITOR_RESULTS_DIR="$DATA/results"' >> ~/.profile
echo 'export JOBMONITOR_METADATA_HOST="34.77.25.246"' >> ~/.profile
echo 'export JOBMONITOR_METADATA_PORT="27017"' >> ~/.profile
echo 'export JOBMONITOR_METADATA_DB="jobmonitor"' >> ~/.profile
echo 'export JOBMONITOR_METADATA_USER="jobmonitor"' >> ~/.profile
echo 'export JOBMONITOR_METADATA_PASS="golden-marine-turtle"' >> ~/.profile
source ~/.profile
