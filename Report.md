
## Describing the Learning Algorithm

This solution uses the same Deep Deterministic Policy Gradient Actor-Critic model as the continuous control project 2. The code for the agent and model are modified from ddpg_agent.py and model.py given in the bipedal example. The model used for both actor and critic networks is a fully connected neural network with two hidden layers with 136 and 56 layers each. The input to the network is the state vector (size 24) and the output is the action space (size 2). This architecture is used for both the local and target networks. 

The agent uses an actor-critic model with an experience replay buffer. The total buffer size is 1E7, the batch size for sampling is 256, the discount factor is 0.99, the soft update tau factor is 0.0015, the learning rate of the critic is 0.004, the learning rate of the actor is 8E-5, and the networks are updated every time step. The actor steps through timesteps and collects the reward/state/action in the experience replay. From the experience buffer, a randomized sample of batch size is used to learn from (which decorrelates temporal data). Every 4 timesteps, the agent learns, meaning the critic network takes the observations and actions to determine Q. The target Q is updated using discount factor, then compared to the local critic network. Adam optimizer and MSE are used. Gradients are clipped for the critic network. Soft update between target and local models happens with tau=0.0015.


## Plot of Rewards

The plot of the rewards at each episode and mean of last 100 episodes is given in the jupyter notebook. 

The number of episodes required for the agent to "solve" the environment is ~7500 episodes. As noted in the benchmark implementation, the episode rewards are noisy, but the mean score does reach 0.5 at ~7500 episodes.

(Note the notebook's error is due to manual keyboard interrupt of training.)

## Ideas for Future Work

Additional work could include altering the actor/critic neural network architectures to include more layers, since the movement of the tennis players may be more abstract than a kinematic motion like continuous control. Also, looking at different noise and/or intialization methods could be interesting, since it seems that the agents would randomly have trouble depending on the seed.
