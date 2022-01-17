# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
The goal of the project is to refactor the model training code found in the jupyter notebook `churn_notebook.ipynb`.
The refactored code needs to follow engineering best practices for implementing software (modular, documented, and tested) and be PEP8 compliant.
It should be possible to run the project from the command line.

## Get started

### Software required
- Docker

### Install dependencies
Build docker image to run the project in
```bash
docker build -t ml-devops .
```
To run interactive bash shell in docker container that has the project dependencies.
```bash
docker run -it -v $(pwd):/ml-devops-1 ml-devops
```

#### Run the model training pipeline
```bash
python3 churn_library.py 
```
This will 
- preprocess the raw data
- Produce EDA plots
- Run the training and evaluation pipeline for both a logistic regression and a random forest model.

#### To run the tests
```bash
python3 churn_script_logging_and_tests.py 
```
You can view the logs from the tests in `logs/churn_library.log`.

## Files in the Repo
```
|-- data                                <- Data folder
|-- docs                                <- Docs folder
|-- images                              <- Folder that contains the plots produces by the exploratory data analysis
|-- logs                                <- Folder that contain logs
|-- results                             <- Folder that contain results from the evaluation of trained models
|-- models                              <- Folder that contains fitted model artifacts                   
|-- model_configs.py                    <- ML model configs 
|-- plotting.py                         <- Plotting functionality
|-- evaluation.py                       <- Functionality for evaluating predictions of a classification model.
|-- churn_library.py                    <- Main script for training the churn models
|-- churn_script_logging_and_tests.py   <- Script for running and loging tests of the code.   
|-- custom_transformers.py              <- Custom sklearn compatible transformers
|-- churn_notebook.ipynb                <- Original notebook to refactor, for reference
|-- requirements.txt                    <- Project python dependencies
|-- Dockerfile                          <- Dockerfile to ensure reproducibility 
`-- README.md                           <- You're here:-)

```
## Choices made
1. The way  the encoding of the categorical variables was happenining in the notebook had the potential to cause data leakage between train and test set, since it was taking the mean of churn in over data sets. I instead implemented a transformer to do it. This way, only the mean from the train set is used.
2. The `train_models` function template in `churn_library.py` was set up to train two models. Instead I made a function `train_model_cross_validation` that takes a model configuration. The function is called twice to produce the results for both models. This way, it would be easy to try a new model without editing the training function. 


