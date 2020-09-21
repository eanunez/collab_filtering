#!/bin/bash

export SETUPTOOLS_USE_DISUTILS=stdlib  # temp solution for broken setuptools == 50
pip3 install virtualenv
# Create virtual environment, venv
virtualenv -p python3 venv
if [ -f $PWD/venv/bin/activate ]; then
    echo   "Load Python virtualenv from '$PWD/venv/bin/activate'"
    source $PWD/venv/bin/activate
    pip3 install -I gunicorn
    pip3 install -r requirements.txt
fi
exec "$@"
### Check if a directory does not exist ###
if [ ! -d "$PWD/dataset" ]; then
    echo "Creating $PWD/dataset" 
    mkdir $PWD/dataset
fi
if [ ! -d "$PWD/model" ]; then
	echo "Creating $PWD/model"
	mkdir $PWD/model
# Initialize model config file
printf "{}" > $PWD/config.json