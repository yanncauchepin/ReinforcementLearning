import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque

env = gym.make('CarRacing-v2', render_mode='rgb_array', lap_complete_percent=0.95, domain_randomize=False, continuous=False)

# Hyperparameters
num_episodes = 1000
max_steps = 200
alpha = 0.001  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
min_epsilon = 0.1
memory_size = 10000

# DQN setup
class DQNAgent:
    def __init__(self, observation_size, action_size):
        self.observation_size = observation_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.epsilon = epsilon
        self.epsilon_min = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.observation_size))
        model.add(tf.keras.layers.MaxPooling2D())
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D())
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return env.action_space.sample()
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # Returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

# Initialize agent
observation_size = env.observation_space.shape
action_size = env.action_space.n
agent = DQNAgent(observation_size, action_size)

# Training loop
for episode in range(num_episodes):
    state, _ = env.reset()
    state = np.reshape(state, [1, 96, 96, 3])
    done = False
    total_reward = 0

    for step in range(max_steps):
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 96, 96, 3])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if done:
            break

    if len(agent.memory) > 32:
        agent.replay(32)

    # Decay epsilon
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

    print(f"Episode {episode + 1}: Total Reward: {total_reward}")

# # Close environment
# env.close()

# def render_environment(env):
#     plt.ion()  # Interactive mode on
#     fig, ax = plt.subplots()

#     for _ in range(10):  # Run 10 episodes for visualization
#         state, _ = env.reset()
#         done = False
        
#         while not done:
#             action = env.action_space.sample()  # Random action for demonstration
#             state, reward, done, _, _ = env.step(action)
#             ax.imshow(state)
#             plt.pause(0.01)  # Pause to update the frame

#     plt.ioff()  # Interactive mode off
#     plt.show()

# # Call the render function
# render_environment(env)