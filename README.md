# Homogeneous-Learning
A self-attention decentralized deep learning model based on reinforcement learning. 

We propose a decentralized learning model called Homogeneous Learning (HL) for tackling non-IID data with a self-attention mechanism. In HL, training performs on each round’s selected node, and then the trained model is sent to the next round’s selected node. Notably, for the selection, the self-attention mechanism leverages reinforcement learning to observe a node’s inner state and its surrounding environment’s state, and find out which node should be selected to optimize the training. 

<img src="fig1.png" width="60%">
