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

ROOT_PATH = Path(__file__).parent


with open(Path(ROOT_PATH, "config.json"), 'r') as f:
    config = json.load(f)
ENVIRONMENT = config["ENV"]["environment"]
LAP_COMPLETE_PERCENT = config["ENV"]["lap_complete_percent"]
DOMAIN_RANDOMIZE = config["ENV"]["domain_randomize"]
CONTINUOUS = config["PPO"]["continuous"]
AGENT_DIR = config["PPO"]["agent_dir"]
LOG_DIR = config["PPO"]["log_dir"]
CHECKPOINT_DIR = config["PPO"]["checkpoint_dir"]

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
AGENT_NOW = f"rlib_{now}"
AGENT_STATIC = "rlib"


ITERATIONS = 10  


os.makedirs(Path(ROOT_PATH, AGENT_DIR), exist_ok=True)
os.makedirs(Path(ROOT_PATH, LOG_DIR), exist_ok=True)
os.makedirs(Path(ROOT_PATH, CHECKPOINT_DIR), exist_ok=True)


if ray.is_initialized():
    ray.shutdown()
ray.init(logging_level="INFO") # "DEBUG" for more verbosity or "ERROR" for less

def train_agent(static=False):
    
    if static is False:
        agent_name = Path(AGENT_NOW)
    else:
        agent_name = Path(AGENT_STATIC)

    config = (
        PPOConfig()
        .environment(
            env=ENVIRONMENT,
            env_config={
                "continuous": CONTINUOUS, 
                "render_mode": None, 
                "lap_complete_percent": LAP_COMPLETE_PERCENT, 
                "domain_randomize": DOMAIN_RANDOMIZE
            }
        )
        .framework("torch")  # or "tf2" for TensorFlow
        .env_runners(num_env_runners=1) 
        .training(
            gamma=0.99,
            lr=0.0003,
            lambda_=0.95,
            train_batch_size=4000, # Adjust based on num_rollout_workers and rollout_fragment_length
            num_epochs=10, # Changed from num_sgd_iter
            vf_loss_coeff=0.5,
            entropy_coeff=0.0,
        )
        .rl_module( # Added for new API stack model configuration
            model_config={
                "conv_filters": [ # Standard CNN structure for 96x96 images
                    [16, [8, 8], 4],
                    [32, [4, 4], 2],
                    [256, [11, 11], 1], 
                ],
                "vf_share_layers": True, # Often good for PPO
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
                "lap_complete_percent": LAP_COMPLETE_PERCENT, 
                "domain_randomize": DOMAIN_RANDOMIZE
            }
        )
        .framework("torch") # or "tf2"
        .env_runners(num_env_runners=0)
        .training()
        .rl_module( # Added for new API stack model configuration
            model_config={ # Ensure model config matches training
                 "conv_filters": [
                    [16, [8, 8], 4],
                    [32, [4, 4], 2],
                    [256, [11, 11], 1],
                ],
                "vf_share_layers": True,
            }
        )
        .resources(num_gpus=0)
    )

    algo = config.build_algo()
    algo.restore(str(agent_path))


    env = gym.make(ENVIRONMENT, continuous=CONTINUOUS, render_mode="human")

    for episode in range(3):
        terminated = truncated = False
        observation, info = env.reset()
        total_reward = 0
        steps = 0
        while not terminated and not truncated:
            action = algo.compute_single_action(observation, explore=False)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
    env.close()
    algo.stop()


if __name__ == "__main__":
    train_agent(static=True)
    agent_overwiew(Path(ROOT_PATH, AGENT_DIR, AGENT_STATIC))