## This is a Tensorflow implementation of the paper [Homogeneous Learning: Self-Attention Decentralized Deep Learning](https://ieeexplore.ieee.org/document/9680704), IEEE Access 2022


## Table of Contents
* [General information](#general-information)
* [Setup instructions](#setup-instructions)
* [Running the systems](#running-the-systems)
* [Further readings](#further-readings)


## General information
Homogeneous Learning (HL) is decentralized neural networks based on the [Global Workspace Theory](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.456.2829&rep=rep1&type=pdf) for fast learning of novel tasks leverging many expert models knowledge. Different from the attention macanism, we leverage reinforcement learning (RL) to generate the meta agent's policy observing its inner state and surrounding environment’s states, such that the systems can quickly adapt to the given tasks. This is the preliminary study of how the human brain can learn new things very fast based on many models of the world.

<img src="architecture.png" width="80%"/>


## Setup instructions
This is a quick guide to get started with the sources. 
### Dependencies 
You will need [Python 3](https://www.python.org/downloads/) and [Tensorflow 2](https://www.tensorflow.org/install/), to run the systems. 

Upgrade pip to the latest version, use:

    sudo python3 -m pip install --upgrade pip
    
Set up other modules and libraries dependencies, use:

    sudo pip3 install -r requirements.txt

## Running the systems
There are two components in HL, the decentralized learning system in the file of **"environment.py"**, and the DQN-based RL agent system in the file of **"node.py"**. More detailed information can be found in the **Section 3.3** of the Homogeneous Learning paper.

**"environment.py"** includes the decentralized learning algorithm, which allows the systems to envolve based on the decisions made by RL agents.

**"node.py"** includes the reinforcement learning algorithm for learning an optimized communication policy based on observations of model parameters and the correlated rewards.

The HL systems can be run from the terminal by simply typing:

    sudo python3 main.py
    
Note that **"main.py"** will include a total of 120 episodes' learning of how to train a local foundation model to achieve a desired goal within the minimum steps, and at the same time with less communication cost, where each episode includes a whole training procedure of the decentralized learning algorithm.

### Making changes
If you want to make changes to the source, such as the total episodes and the training goal, you are going to need to refer to the **Section 4.1, 4.2.1, A.2** in the paper for more information on how these components work with each other.

## Citation
If this repository is helpful for your research or you want to refer the provided results in this work, you could cite the work using the following BibTeX entry:

```
@article{sun2022homolearn,
  author    = {Yuwei Sun and
               Hideya Ochiai},
  title     = {Homogeneous Learning: Self-Attention Decentralized Deep Learning},
  journal   = {IEEE Access},
  year      = {2021}
}
```

## Further readings
### Global Workspace Theory
* [The Consciousness Prior](https://arxiv.org/abs/1709.08568), Yoshua Bengio, arXiv preprint.
* [Coordination among neural modules through a shared global workspace](https://arxiv.org/abs/2103.01197), Goyal et al., ICLR'22.
* [GFlowNet Foundations](https://arxiv.org/abs/2111.09266), Bengio et al., arXiv preprint.

### Decentralized ML
* [Decentralized Deep Learning for Multi-Access Edge Computing: A Survey on Communication Efficiency and Trustworthiness](https://www.techrxiv.org/articles/preprint/Decentralized_Deep_Learning_for_Multi-Access_Edge_Computing_A_Survey_on_Communication_Efficiency_and_Trustworthiness/16691230), Yuwei Sun et al., IEEE Transactions on Artificial Intelligence.  
