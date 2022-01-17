mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))

run_and_log_tests:
	python3 run_and_log_tests.py > logs/churn_library.log

########################################################
# Utils
########################################################

build:
	docker build -t ml-devops .

dev_docker:
	docker run -it -v $(mkfile_dir):/ml-devops-1 ml-devops
