"""
==========================================
Q-Learning toy sample based on OpenAI Gym.
==========================================

Source: 
 * https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
 * https://github.com/GaurangSharma18/Reinforcement-Learning-Open-AI-gym---Taxi
 * Sutton Book, "Reinforcement Learning"

In RL, QLearning, we want to learn a Q table, a function, for which given a 
state we can check the all the different actions to determine which one has
the greater value on the Q table, and pick that one.  It is very easy to 
implement, works very well, but it requires a lot of episodes.
This Q table can be replaced by a neural network (as function approximator).

5x5 grid
4 destinations
4+1 passengers locations (the 4 destinations plus inside the taxi)

+---------+
|R: | : :G|
| : | : : |
| : : : : |
| | : | : |
|Y| : |B: |
+---------+

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
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make('Taxi-v3', render_mode="ansi").env

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

while (not done) and (epochs<400):
    action = env.action_space.sample()
    #print(f"Action: {action}")
    state, reward, done, info,ext = env.step(action)
    #print(f"State: {state}")


    if reward == -10:
        penalties += 1
    
    # Put each rendered frame into dict for animation
    frames.append({
        'frame': env.render(),
        'state': state,
        'action': action,
        'reward': reward
        }
    )

    epochs += 1

     
# Let's see what is happening..... this is 100% exploration without any learning.
from time import sleep, time

def print_frames(frames):
    for i, frame in enumerate(frames):
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)
        print(chr(27) + "[2J")
        
print_frames(frames)


print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))


# Q-Learning rule, Q(s,a) <- (1-𝛂) Q(s,a) + 𝛂 ( R + γ max_an Q(next state, an)
# Alpha is the explotation-exploration tradeoff whle gamma is the greedy-longterm term (0 is greedy)
env.reset()

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

q_table = np.zeros([5 * 5 * 5 * 4 , env.action_space.n])

reward_list = []
ave_reward_list = []

tot_rewards = 0

for i in range(1, 100000):
    env.reset()
    done = False
    state=0
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state,:]) # Exploit learned values

        #print(f"Action: {action}")
        next_state, reward, done, info,ext = env.step(action)
        #print(f"State: {state} -> {next_state}")

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state,:])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        tot_rewards += reward
        if reward == -10:
            penalties += 1
        
        # Put each rendered frame into dict for animation
        frames.append({
            'frame': env.render(),
            'state': state,
            'action': action,
            'reward': reward
            }
        )

        epochs += 1
        state=next_state

    # Track rewards
    reward_list.append(tot_rewards)
    penalties=0
    tot_rewards=0
    
    if (i+1) % 100 == 0:
        ave_reward = np.mean(reward_list)
        ave_reward_list.append(ave_reward)
        reward_list = []
        
    if (i+1) % 100 == 0:    
        print('Episode {} Average Reward: {}'.format(i+1, ave_reward))

plt.plot(100*(np.arange(len(ave_reward_list)) + 1), ave_reward_list)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Episodes')
#plt.savefig('rewards.pdf')     
#plt.close()  
plt.show()
print("Training finished.\n")



"""Evaluate agent's performance after Q-learning"""

total_epochs, total_penalties = 0, 0
episodes = 100
total_penalties_optimal_list = []

for _ in range(episodes):
    env.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    state=0
    while not done:
        action = env.action_space.sample()
        state, reward, done, info, ext = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs
    total_penalties_optimal_list.append(penalties)

print(f"Random Actions")
print(f"--------------")
print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")


total_epochs, total_penalties = 0, 0
episodes = 100
total_penalties_random_list = []

for _ in range(episodes):
    env.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    state=0
    while not done:
        action = np.argmax(q_table[state,:])
        state, reward, done, info, ext = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs
    total_penalties_random_list.append(penalties)

print(f"Optimal Policy")
print(f"--------------")
print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")



plt.plot(total_penalties_random_list, label='Random Actions')
plt.plot(total_penalties_optimal_list, label='Optimal Policy')
plt.xlabel('Episodes')
plt.ylabel('Penalties')
plt.title('Penalties vs Episodes')
plt.show()

done = False
frames = [] # for animation
env.reset()
while not done:
    action = np.argmax(q_table[state,:])
    state, reward, done, info, ext = env.step(action)

    frames.append({
        'frame': env.render(),
        'state': state,
        'action': action,
        'reward': reward
        }
    )

    epochs += 1

print_frames(frames)