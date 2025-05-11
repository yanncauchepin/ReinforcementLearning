import ray # Use torch or tensoflow as the backend
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
import gymnasium as gym
import os
import shutil
from pathlib import Path
import datetime
from tqdm import tqdm
import json
import numpy as np
import torch

ROOT_PATH = Path(__file__).parent


with open(Path(ROOT_PATH, "config.json"), 'r') as f:
    config = json.load(f)
ENVIRONMENT = config["ENV"]["environment"]
CONTINUOUS = config["PPO"]["continuous"]
GRAVITY = config["ENV"]["gravity"]
ENABLE_WIND = config["ENV"]["enable_wind"]
WIND_POWER = config["ENV"]["wind_power"]
TURBULENCE_POWER = config["ENV"]["turbulence_power"]
AGENT_DIR = config["PPO"]["agent_dir"]
LOG_DIR = config["PPO"]["log_dir"]
CHECKPOINT_DIR = config["PPO"]["checkpoint_dir"]

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
AGENT_NOW = f"rlib_{now}"
AGENT_STATIC = "rlib"


ITERATIONS = 100
GAMMA = 0.99
LEARNING_RATE = 0.0003 
LAMBDA = 0.95
TRAIN_BATCH_SIZE = 4000
NUM_EPOCHS = 10
VF_LOSS_COEFF = 0.5
ENTROPY_COEFF = 0.0  


os.makedirs(Path(ROOT_PATH, AGENT_DIR), exist_ok=True)
os.makedirs(Path(ROOT_PATH, LOG_DIR), exist_ok=True)
os.makedirs(Path(ROOT_PATH, CHECKPOINT_DIR), exist_ok=True)


if ray.is_initialized():
    ray.shutdown()
ray.init(logging_level="INFO") # "DEBUG" for more verbosity or "ERROR" for less

def train_agent(static=False):
    
    if static is False:
        agent_name = AGENT_NOW
    else:
        agent_name = AGENT_STATIC

    config = (
        PPOConfig()
        .environment(
            env=ENVIRONMENT,
            env_config={
                'render_mode': None,  
                'continuous': CONTINUOUS,
                'gravity': GRAVITY,
                'enable_wind': ENABLE_WIND,
                'wind_power': WIND_POWER,
                'turbulence_power': TURBULENCE_POWER,
            }
        )
        .framework("torch")  # or "tf2" for TensorFlow
        .env_runners(num_env_runners=4) 
        .training(
            gamma=GAMMA,
            lr=LEARNING_RATE,
            lambda_=LAMBDA,
            train_batch_size=TRAIN_BATCH_SIZE, 
            num_epochs=NUM_EPOCHS, 
            vf_loss_coeff=VF_LOSS_COEFF,
            entropy_coeff=ENTROPY_COEFF
        )
        .rl_module(
            model_config={
                "fcnet_hiddens": [256, 256],  
                "fcnet_activation": "tanh",   
                "vf_share_layers": True, 
            }
        )
        .resources(num_gpus=1)
    )

    algo = config.build_algo()

    for i in tqdm(range(ITERATIONS), desc="Training"):
        result = algo.train()
        print(f"Iteration: {i + 1}")
        print(pretty_print(result))

        if (i + 1) % 5 == 0: # Checkpoint every 5 iterations
            algo.save(Path(ROOT_PATH, CHECKPOINT_DIR, agent_name, str(i)))

    trained_agent = algo.save(Path(ROOT_PATH, AGENT_DIR, agent_name))
    algo.stop()
    
    return trained_agent


def agent_overwiew(agent_path):

    config = (
        PPOConfig()
        .environment(
            env=ENVIRONMENT,
            env_config={
                "continuous": CONTINUOUS, 
                "render_mode": "human", 
            }
        )
        .framework("torch") # or "tf2"
        .env_runners(num_env_runners=0)
        .training()
        .rl_module(
            model_config={ 
                "fcnet_hiddens": [256, 256],  
                "fcnet_activation": "tanh",   
                "vf_share_layers": True, 
            }
        )
        .resources(num_gpus=1)
    )

    algo = config.build_algo()
    algo.restore(str(agent_path))

    module = algo.get_module()
    module.eval()


    env = gym.make(ENVIRONMENT, continuous=CONTINUOUS, render_mode="human")

    for episode in range(3):
        terminated = truncated = False
        observation, info = env.reset()
        total_reward = 0
        steps = 0
        while not terminated and not truncated:
            if not isinstance(observation, np.ndarray):
                observation = np.array(observation)
            observation_tensor = torch.tensor(observation, dtype=torch.float32)
            with torch.no_grad():
                action_output = module.forward_inference({"obs": observation_tensor})
            action = action_output["action_dist_inputs"].cpu().numpy()
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
    env.close()
    algo.stop()


if __name__ == "__main__":
    # train_agent(static=True)
    agent_overwiew(Path(ROOT_PATH, AGENT_DIR, AGENT_STATIC))