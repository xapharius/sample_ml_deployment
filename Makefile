SHELL := /bin/bash

PROD_FOLDER := ./deployment
# assumes conda enabled via new conda.sh when calling make
CONDA_PATH := $(CONDA_PREFIX)

.PHONY: prod_venv_install prod_venv_export dev_venv_instal dev_venv_export download_data


prod_venv_install:
	conda create --file $(PROD_FOLDER)/prod_requirements.txt --prefix ./deployment/venv

prod_venv_export:
	. $(CONDA_PATH)/etc/profile.d/conda.sh && \
	conda activate $(PROD_FOLDER)/venv && \
	conda list --explicit > $(PROD_FOLDER)/prod_requirements.txt

dev_venv_install:
	conda create --file ./dev_requirements.txt --prefix ./venv

dev_venv_export:
	. $(CONDA_PATH)/etc/profile.d/conda.sh && \
	conda activate ./venv && \
	conda list --explicit > ./dev_requirements.txt

download_data:
	curl --create-dirs -o ./data/titanic.csv https://raw.githubusercontent.com/pcsanwald/kaggle-titanic/master/train.csv 

