#!/usr/bin/zsh
#===============================================================================
# Create conda env for vot and install necessary parts
# run the script via source init_vot.sh
#===============================================================================
CONDA_ENV="vot"

eval "$(conda shell.bash hook)"

#check if CONDA_ENV already exists and create it if not
ENVS=$(conda env list | awk '{print $1}' )
if [[ $ENVS = *"$CONDA_ENV"* ]]; then
    echo "Conda env found"
    conda activate ${CONDA_ENV}
else
    echo "Conda env not found, creating new one"
    echo "yes" | conda create -n vot python=3.6
    conda activate vot
fi;

pip install git+https://github.com/votchallenge/vot-toolkit-python

vot initialize vot2019 --workspace vot2019
