# Homogeneous-Learning
> A self-attention decentralized deep learning algorithm based on reinforcement learning.


## Table of Contents
* [General information](#general-information)
* [System architecture](#system-architecture)
* [Setup instructions](#setup-instructions)
* [Running the systems](#running-the-systems)


## General information
Federated learning (FL) has been facilitating privacy-preserving deep learning in many walks of life such as medical image classification, network intrusion detection, and so forth. Whereas it necessitates a central parameter server for model aggregation, which brings about delayed model communication and vulnerability to adversarial attacks. A fully decentralized architecture like Swarm Learning allows peer-to-peer communication among distributed nodes, without the central server. One of the most challenging issues in decentralized deep learning is that data owned by each node are usually non-independent and identically distributed (non-IID), causing time-consuming convergence of model training.　

Homogeneous Learning (HL) is a decentralized learning model for tackling non-IID data with a self-attention mechanism. In HL, training performs on each round’s selected node, and then the trained model is sent and further processed by the next round’s selected node. Notably, for the selection, the self-attention mechanism leverages reinforcement learning (RL) to observe a node’s inner state and its surrounding environment’s state, thus finding out which one should be selected at each step to optimize the training.


## System architecture
HL leverages RL agents to learn a shared optimized communication policy in the inner loop of model training, thus contributing to a fast convergence of training and reduced communication cost. Each node has two machine learning (ML) models, i.e., a node model which we call the local foundation model for a specific ML task and a RL model for the decision-making of peer-to-peer communications.

<img src="architecture.png" width="50%"/>


## Setup instructions
This is a quick guide to get started with the sources. 
### Dependencies 
You will need [Python 3](https://www.python.org/downloads/) and [Tensorflow 2](https://www.tensorflow.org/install/), to run the systems. 

Set up other modules and libraries dependencies, use:

    $ sudo pip3 install -r requirements.txt


### Forking or cloning
Consider ***forking*** the project if you want to make changes to the sources. If you simply want to run it locally, you can simply ***clone*** it.

#### Forking
If you decide to fork, follow the [instructions](https://help.github.com/articles/fork-a-repo) given by github. After that you can clone your own copy of the sources with:

    $ git clone https://github.com/YOUR_USER_NAME/homogeneous-learning.git

Make sure you change *YOUR_USER_NAME* to your user name.

## Running the systems
There are two components in HL, the decentralized learning system in the file of <strong>"environment.py"</strong>, and the DQN-based RL agent system in the file of <strong>"node.py"</strong>. More detailed information can be found in the Section 3.3 of the Homogeneous Learning paper.

<strong>"environment.py"</strong> includes the decentralized learning algorithm, which allows the systems to envolve based on the decisions made by RL agents.

<strong>"node.py"</strong> includes the reinforcement learning algorithm for learning an optimized communication policy based on observations of model parameters and the correlated rewards.

The HL systems can be run from the terminal by simply typing:

    $ sudo python3 main.py
    
Note that <strong>"main.py"</strong> will include a total of 120 episodes' learning of how to train a local foundation model to achieve a desired goal within the minimum steps, and at the same time with less communication cost, where each episode includes a whole training procedure of the decentralized learning algorithm.

### Making changes
If you want to make changes to the source, such as the total episodes and the training goal, you are going to need to refer to the Section 4.1, 4.2.1, A.2 in the paper for more information on how these components work with each other.




