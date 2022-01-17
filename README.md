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
This will run the training and evaluation pipeline for both a logistic regression and a random forest model.
The evaluation metrics and model artifacts can be found in 
- `modelling_artifacts/logreg` for the logistic regression model 
- `modelling_artifacts/random_forest` for the random forest model

To run the tests
```bash
make run_and_log_tests
```
You can view the logs from the tests in `logs/churn_library.log`.

## Project organization
```
|-- requirements.txt                <- Python dependencies
|-- setup.py                        <- For installing project as package
|-- Dockerfile                      <- For installing project as package
|-- Makefile                        <- Run individual parts of the pipeline or utility commands.
|-- README.md
|-- run_and_log_tests.py            <- Run tests and log the output
|-- data                            <- Data folder
|-- docs                            <- Documentation folder
|-- eda                             <- Output from explorative data analysis
|-- modelling_artifacts             <- Output from model training and evaluatiuon
|-- src                             <- Source code for the project
|   |-- __init__.py
|   |-- config.py                   <- Configuration of project
|   |-- logger.py                   <- Setup of logger
|   |-- data                        <- Code related to data transformations
|   |   |-- __init__.py
|   |   |-- feature_engineering.py  <- Code related to data transformations
|   |   `-- preprocessing.py        <- Feature engineering code
|   `-- modelling
|       |-- __init__.py
|       |-- custom_transformers.py  <- Custom sklearn compatible transformers
|       |-- eda.py                  <- Exploratory data analysis code
|       |-- evaluation.py           <- Evauation of model performance
|       |-- model_configs.py        <- Confguration of models
|       `-- train_evaluate.py       <- Training and evaluation of models
`-- logs
    `-- churn_library.log           <- Log of tests
```

## Choices
The way that the encoding of the categorical variables - replacing them with the mean of the churn in the category - was happeining in the notebook had the potential to cause data leakage between train and test set, since it whas taking the mean on both.
I instead implemented a transformer to do it. This way, only the mean from the train set is used. 


