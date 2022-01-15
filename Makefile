preprocess:
	python preprocessing.py

interactive_docker:
	docker run -it -v /Users/jesravnbol/code/ml-devops-1:/ml-devops-1 ml-devops

build:
	docker build -t ml-devops .
