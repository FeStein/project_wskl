# Script is use to quickly setup a working conda environment to use this project

CONDA_ENV_NAME=test

conda create --name $CONDA_ENV_NAME python=3.9 -y 

conda activate $CONDA_ENV_NAME 

conda install -c conda-forge opencv


