# Homogeneous-Learning
> A self-attention decentralized deep learning algorithm based on reinforcement learning.


## Table of Contents
* [Background](#background)
* [General Info](#general-information)
* [System Architecture](#system-architecture)
* [Setup](#setup)
* [Usage](#usage)


## Background
Federated learning (FL) has been facilitating privacy-preserving deep learning in many walks of life such as medical image classification, network intrusion detection, and so forth. Whereas it necessitates a central parameter server for model aggregation, which brings about delayed model communication and vulnerability to adversarial attacks. A fully decentralized architecture like Swarm Learning allows peer-to-peer communication among distributed nodes, without the central server. One of the most challenging issues in decentralized deep learning is that data owned by each node are usually non-independent and identically distributed (non-IID), causing time-consuming convergence of model training.　


## General Information
Homogeneous Learning (HL) is a decentralized learning model for tackling non-IID data with a self-attention mechanism. In HL, training performs on each round’s selected node, and then the trained model is sent and further processed by the next round’s selected node. Notably, for the selection, the self-attention mechanism leverages reinforcement learning (RL) to observe a node’s inner state and its surrounding environment’s state, thus finding out which one should be selected at each step to optimize the training.


## System Architecture
HL leverages RL agents to learn a shared optimized communication policy in the inner loop of model training, thus contributing to a fast convergence of training and reduced communication cost. Each node has two machine learning (ML) models, i.e., a node model which we call the local foundation model for a specific ML task and a RL model for the decision-making of peer-to-peer communications.

<img src="fig1.png" width="50%"/>


## Setup
This folder contains three python files, <strong>"environment.py"</strong>, <strong>"node.py"</strong>, and <strong>"main.py"</strong>. 

<strong>"environment.py"</strong> includes the decentralized learning algorithm, which allows the systems to envolve based on the decisions made by RL agents.

<strong>"node.py"</strong> includes the reinforcement learning algorithm for learning an optimized communication policy based on observations of model parameters and the correlated rewards.

<strong>"main.py"</strong> includes a total of 120 episodes' learning of how to train a local foundation model to achieve a desired goal within the minimum steps and with less communication cost. Each episode consists of a whole training process of the decentralized learning algorithm.

To start using Homogeneous Learning:
`
python3 main.py
`


## Usage




