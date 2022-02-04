
## Describing the Learning Algorithm

This solution uses the Deep Deterministic Policy Gradient Actor-Critic Network. The code for the agent and model are modified from ddpg_agent.py and model.py given in the bipedal example. The model used for both actor and critic networks is a fully connected neural network with two hidden layers with 256 and 126 layers each. The input to the network is the state vector (size 33) and the output is the action space (size 4). This architecture is used for both the local and target networks. 

The agent uses an actor-critic model with an experience replay buffer. The total buffer size is 1E8, the batch size for sampling is 500, the discount factor is 0.99, the soft update tau factor is 0.0021, the learning rate of the critic is 0.001, the learning rate of the actor is 8E-4, and the networks are updated every 4 time steps. The actor steps through timesteps and collects the reward/state/action in the experience replay. From the experience buffer, a randomized sample of batch size is used to learn from (which decorrelates temporal data). Every 4 timesteps, the agent learns, meaning the critic network takes the observations and actions to determine Q. The target Q is updated using discount factor, then compared to the local critic network. Adam optimizer and MSE are used. Gradients are clipped for the critic network. Soft update between target and local models happens with tau=0.0021.


## Plot of Rewards

The plot of the rewards at each episode is given below. 

The number of episodes required for the agent to "solve" the environment is ~200 episodes.

(reward_plot.jpg)


## Ideas for Future Work

Additional work could include altering the actor/critic neural network architectures to include LSTMs, since the motion of the orbs are very predictable once the episode starts. Also, looking at different noise and/or intialization methods could be interesting, since it seems that the agents would randomly have trouble depending on the seed.