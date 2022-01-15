mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))

modelling_pipeline: preprocess add_features eda train_evaluate_logreg

preprocess:
	python3 src/data/preprocessing.py

add_features:
	python3 src/data/feature_engineering.py

eda:
	python3 src/modelling/eda.py

train_evaluate_logreg:
	python3 src/modelling/train_evaluate.py "src.modelling.model_configs.logreg" "logreg"



########################################################
# Utils
########################################################

build:
	docker build -t ml-devops -f ./Dockerfile_dev .

dev_docker:
	docker run -it -v $(mkfile_dir):/ml-devops-1 ml-devops
