mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))

build:
	docker build -t ml-devops -f ./Dockerfile_dev .

dev_docker:
	docker run -it -v $(mkfile_dir):/ml-devops-1 ml-devops

preprocess:
	python3 src/data/preprocessing.py

add_features:
	python3 src/data/feature_engineering.py

train:
	python3 src/modelling/train_evaluate.py

