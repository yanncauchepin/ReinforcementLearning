import numpy as np
import random
import tensorflow as tf
from collections import deque
import os

from env import ENV

observation_size = ENV.observation_space.shape
action_size = ENV.action_space.n

class DQNAgent():

    def __init__(
            self, 
            artefact_name,
            num_episodes = 10000,
            max_steps = 200,
            alpha = 0.001,
            gamma = 0.99,
            epsilon = 1.0, 
            epsilon_decay = 0.9995,
            min_epsilon = 0.1,
            memory_size = 10000
            ):
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)
        self.artefact_name = artefact_name
        self.model = self.build_model()
        with open(f'trained_agents/{self.artefact_name}_log.csv', 'w') as log:
            log.write("episode, steps, reward")

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=observation_size))
        model.add(tf.keras.layers.MaxPooling2D())
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D())
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(action_size, activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return ENV.action_space.sample()
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # Returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def save_info(self):
        with open(f'info_{self.artefact_name}.txt', 'w') as info:
            info.write(f"NUM EPISODE: {self.num_episodes}")
            info.write(f"MAX_STEPS: {self.max_steps}")
            info.write(f"ALPHA: {self.alpha}")
            info.write(f"GAMMA: {self.gamma}")
            info.write(f"EPSILON: {self.epsilon}")
            info.write(f"EPSILON_DECAY: {self.epsilon_decay}")
            info.write(f"MIN_EPSILON: {self.min_epsilon}")
            info.write(f"MEMORY_SIZE: {self.memory_size}")
    
    def save_artefact(self, episode, steps, reward):
        self.model.save(f"trained_agents/{self.artefact_name}.keras")
        with open(f'trained_agents/{self.artefact_name}_log.csv', 'a') as log:
            log.write(f"\n{episode}, {steps}, {reward}")

    def train(self):
        for episode in range(self.num_episodes):
            state, _ = ENV.reset()
            state = np.reshape(state, [1, 96, 96, 3])
            done = False
            total_reward = 0

            for step in range(self.max_steps):
                action = self.act(state)
                next_state, reward, done, _, _ = ENV.step(action)
                next_state = np.reshape(next_state, [1, 96, 96, 3])
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if done:
                    break
            
            self.save_artefact(episode, step, total_reward)
            
            if len(self.memory) > 32:
                self.replay(32)

            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay

            print(f"Episode {episode}: Total Reward: {total_reward}")
        
        
        


