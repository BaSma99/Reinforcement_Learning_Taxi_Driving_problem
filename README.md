# Reinforcement Learning: Individual Assignment

## Name: Basma Reda Shaban Abd-Elsalam Abd-Elwahab

## Email: babde014@uottawa.ca


## Dependanceis:
- Install the gym library in python
```python
!pip install cmake 'gym[atari]' scipy
import gym
```
 ## Defining the Agent:
 ![](https://storage.googleapis.com/lds-media/documents/Reinforcement-Learning-Animation.gif)

## List the important definations of an Agent:
 
### 1- Rewards:
 - The agent should receive a high positive reward for a successful dropoff because this behavior is highly desired

### 2- State space:
- The State Space is the set of all possible situations our taxi could inhabit.
![](https://storage.googleapis.com/lds-media/images/Reinforcement_Learning_Taxi_Env.width-1200.png)

### 3- Action space:
- The set of all the actions that our agent can take in a given state.

## Defining Our Problem:
- Make some modification in the following code https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/ to : 
  - Turn this code into a module of functions that can use multiple environments
  - Tune alpha, gamma, and/or epsilon using a decay over episodes
  - Implement a grid search to discover the best hyperparameters

# Implemeting our problem with Python:

## Step 1: Importing important libraries
```python
from numpy import array #to deal with arrays
import math #to deal with math functions
from math import inf
from numpy.linalg import norm
import gym #for developing reinforcement learning environments
import random 
import numpy as np
from IPython.display import clear_output #to produce clear output
from time import sleep #to sleep for a time 
```

## Step 2: Define our reinforcement learning environment:
```python
env = gym.make("Taxi-v3")
```

## Step 3: Turning the code into a module of functions that can use multiple environments:
```python
def set_time(env): #to set the time estimated
def print_frames(frames): #to print our environment frames
def TrainAgent(env): #to train the agnet
def launch_game(q_table, env): # to evaluate and launch the agent
```

- The result before hyperparameters tuning is:
    - Average timesteps per episode: 300.33
    - Average penalties per episode: 0.0

## Step 4: Tuning alpha, gamma, and/or epsilon using a decay over episodes:
```python
new_env1 = gym.make("Taxi-v3")
new_q_table_1 = HyperParametersTuning(new_env1,0.1,0.1,0.1)
launch_game(new_q_table_1,new_env1)
```

- The result after hyperparameters tuning is:
   - Episode: 15000
   - Average timesteps per episode: 20100.0
   - Average penalties per episode: 0.0
## Step 5: Implementing a grid search to discover the best hyperparameters:
```python
def TrainingGridSearch(env,epsilon, alpha, gamma):
def EvaluateGridSearch(q_table, env):
def grid_search(parm,env): 
```
```python
def grid_search(parm,env):
  time_steps = 15000
  penalties = 15000
  GridSearchHyperParameters = parm
  for i in GridSearchHyperParameters['alpha']:
    for j in GridSearchHyperParameters['gamma']:
      for k in GridSearchHyperParameters['epsilon']:
        q_table,alpha,gamma,epsilon = TrainingGridSearch(env,alpha=i,gamma=j,epsilon=k)
        TAvg,pAvg = EvaluateGridSearch(q_table,env)
        if TAvg<= time_steps:
          if pAvg <= penalties:
            time_steps = TAvg
            penalties = pAvg
            bestparameter = {'alpha':alpha,'gamma':gamma,'epsilon':epsilon,'Time':TAvg,'penalties':pAvg}
  return bestparameter
```

- The result after grid search is:
   - Episode: 15000
   - Average timesteps per episode: 696.65
   - Average penalties per episode: 0.0
   - {'Time': 289.95, 'alpha': 0.1, 'epsilon': 0.6, 'gamma': 0.1, 'penalties': 0.0}

## Step 6: Apply the training and evaluation agent functions to a different Q learning environment called 'FrozenLake-v0' environment
 - Train and evaluate the FrozenLake-v0 environment
```python
env2 = gym.make("FrozenLake-v0")
q_table2 = TrainAgent(env2)
launch_game(q_table2, env2)
```

- The result is: 
   - Average timesteps per episode: 276.52
   - Average penalties per episode: 0.0


- Tuning the hyperparameters of FrozenLake-v0 environment:
```python
new_env2 = gym.make("FrozenLake-v0")
new_q_table_2 = HyperParametersTuning(new_env2,0.1,0.1,0.1)
launch_game(new_q_table_2,new_env2)
```

- Apply Grid search to discover the best hyperparameters:
```python
GridSearchHyperParameters = {'alpha':[0.1,0.2,0.3],'gamma':[0.1,0.2,0.3],'epsilon':[0.1,0.2,0.3]}
grid_search(GridSearchHyperParameters,env2)
```
- The result is:
    - Episode: 15000
    - Average timesteps per episode: 204.12
    - Average penalties per episode: 0.0
    - {'Time': 187.38, 'alpha': 0.1, 'epsilon': 0.6, 'gamma': 0.1, 'penalties': 0.0}