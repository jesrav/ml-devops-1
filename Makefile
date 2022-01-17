mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))

all: preprocess add_features eda
all: train_evaluate_logreg train_evaluate_random_forest

preprocess:
	python3 src/data/preprocessing.py

add_features:
	python3 src/data/feature_engineering.py

perform_eda:
	python3 src/modelling/eda.py

train_evaluate_logreg:
	python3 src/modelling/train_evaluate.py \
	"LogregConfig" \
	"logreg" \
	"modelling_artifacts/logreg"

train_evaluate_random_forest:
	python3 src/modelling/train_evaluate.py \
	"RandomForestConfig" \
	"random_forest" \
	"modelling_artifacts/random_forest"

run_and_log_tests:
	python3 tests/churn_script_logging_and_tests.py > logs/churn_library.log

########################################################
# Utils
########################################################

build:
	docker build -t ml-devops .

dev_docker:
	docker run -it -v $(mkfile_dir):/ml-devops-1 ml-devops
