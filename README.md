# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
The projects is the 1 part of the ML DevOps Engineer Nanodegree.
The goal was to refactor the contents of a jupyter notebook.

## Get started

### Software required
- Docker

### Install dependencies
Build docker image to run the project in
```bash
docker build -t ml-devops .
```

### Run ml pipeline
from the project root run an interactive bash shell in a docker container that has the project dependencies.
```bash
docker run -it -v $(pwd):/ml-devops-1 ml-devops
```

Run the model training pipeline
```bash
make all
```

Run the tests
```bash
make run_and_log_tests
```
You can view the logs from the tests in `logs/churn_library.log`.

# Project organization

