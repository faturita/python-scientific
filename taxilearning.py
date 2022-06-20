"""
==========================================
Q-Learning toy sample based on OpenAI Gym.
==========================================

Source: https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
        Sutton Book, "Reinforcement Learning"

In RL, QLearning, we want to learn a Q table, a function, for which given a state we can check the all the different actions to determine which one has
the greater value on the Q table, and pick that one.  It is very easy to implement, works very well, but it requires a lot of episodes.
This Q table can be replaced by a neural network (as function approximator).

5x5 grid
4 destinations
4+1 passengers locations (the 4 destinations plus inside the taxi)

6 possible actions

0 south
1 north
2 east
3 west
4 pickup
5 dropoff

5 x 5 x 5 x 4  states

"""

import gym

env = gym.make('Taxi-v3').env

env.render()

env.reset()
env.render()

print('Action state {}'. format(env.action_space))
print('State space {}'.format(env.observation_space))

state = env.encode(3,1,2,0)    # Taxi location row and column, 2 is where the passenger is located (the second location), and 0 is where it should go.

env.s = state
env.render()

# env.P represents {action: prob, nextstate, rewards, done} 
# When done is true, the episode finishes and we had succeeded.
print(env.P[state])

env.s = 328  # set environment to illustration's state

epochs = 0
penalties, reward = 0, 0

frames = [] # for animation

done = False

while not done:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

    if reward == -10:
        penalties += 1
    
    # Put each rendered frame into dict for animation
    frames.append({
        'frame': env.render(mode='ansi'),
        'state': state,
        'action': action,
        'reward': reward
        }
    )

    epochs += 1
    
    
print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))


# Let's see what is happening..... this is 100% exploration without any learning.
#from IPython.display import clear_output
from time import sleep, time

def print_frames(frames):
    for i, frame in enumerate(frames):
        #clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)
        
print_frames(frames)


# QLEarning rule, Q(s,a) <- (1-𝛂) Q(s,a) + 𝛂 ( R + γ max_an Q(next state, an)
# Alpha is the explotation-exploration tradeoff whle gamma is the greedy-longterm term (0 is greedy)

import numpy as np
q_table = np.zeros([env.observation_space.n, env.action_space.n])

import random

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# For plotting metrics
all_epochs = []
all_penalties = []

for i in range(1, 100001):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values

        next_state, reward, done, info = env.step(action) 
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1
        
    if i % 100 == 0:
        #clear_output(wait=True)
        print(f"Episode: {i}")

print("Training finished.\n")

"""Evaluate agent's performance after Q-learning"""

total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")
