# ECE-472

This repo contains projects from **ECE-472 Deep Learning** course of **The Cooper Union**.

All rights reserved to **Youngwoong Cho** (Cooper Union ME&CS '23)

# Project Desccription
- `./hw1/`

    Perform linear regression of a noisy sinewave using a set of gaussian basis functions with learned location and scale parameters. Model parameters are learned with stochastic gradient descent. Use of automatic differentiation is required.
- `./hw2/`

    Perform binary classification on the spirals dataset using a multi-layer perceptron. You must generate the data yourself.    
- `./hw3/`

    Classify MNIST digits with a (optionally convoultional) neural network. Get at least 95.5% accuracy on the test test.
- `./hw4/`

    Classify CIFAR10. Acheive performance similar to the state of the art. Classify CIFAR100. Achieve a top-5 accuracy of 90%.
- `./hw5/`

    Classify the AG News dataset.
- `./midterm/`

    The goal of the midterm project is to reproduce results from a contemporary research paper.
    
# How to run
## Run within Docker container
1. Run following command: `tf.sh python3 {project_name}/main.py`.

>`{project_name}` is one of the followings:
> - `hw1`
> - `hw2`
> - `hw3`
> - `hw4`
> - `hw5`
> - `midterm`

2. If you have trouble running docker image, update `./Dockerfile`.

## Run locally
1. This project was tested with `python==3.8.0`.
2. Install dependencies: `pip install -r requirements.txt`
3. Run following command: `python3 {project_name}/main.py`.

>`{project_name}` is one of the followings:
> - `hw1`
> - `hw2`
> - `hw3`
> - `hw4`
> - `hw5`
> - `midterm`

# Troubleshooting
If you havee trouble running the code, please contact <cho4@cooper.edu> or <herocho1997@gmail.com>.
