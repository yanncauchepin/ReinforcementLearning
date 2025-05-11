import gymnasium as gym
from stable_baselines3 import PPO # Use torch as the backend
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
import os
from pathlib import Path
import datetime
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
AGENT_NOW = f"sd3_{now}"
AGENT_STATIC = "sd3"

TIMESTEPS = 50000
N_STEPS = 3000
LEARNING_RATE = 0.0003
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENT_COEF = 0.0

os.makedirs(Path(ROOT_PATH, AGENT_DIR), exist_ok=True)
os.makedirs(Path(ROOT_PATH, LOG_DIR), exist_ok=True)
os.makedirs(Path(ROOT_PATH, CHECKPOINT_DIR), exist_ok=True)

def train_agent(static=False):
    
    if static:
        agent_name = AGENT_STATIC
    else:
        agent_name = AGENT_NOW

    agent_path = Path(ROOT_PATH, AGENT_DIR, agent_name)
    log_path = Path(ROOT_PATH, LOG_DIR, agent_name)

    env_kwargs = {
        'continuous': CONTINUOUS,
        'lap_complete_percent': LAP_COMPLETE_PERCENT,
        'domain_randomize': DOMAIN_RANDOMIZE,
        'render_mode': None
    }
    vec_env = make_vec_env(ENVIRONMENT, n_envs=1, env_kwargs=env_kwargs)
    vec_env = VecFrameStack(vec_env, n_stack=4)

    # "CnnPolicy" is used because the observation space is an image.
    agent = PPO(
        "CnnPolicy", 
        vec_env, 
        verbose=1, 
        tensorboard_log=log_path,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS, 
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA, 
        ent_coef=ENT_COEF,
        device="cuda"
    )

    agent.learn(total_timesteps=TIMESTEPS, progress_bar=True)

    agent.save(agent_path)

    vec_env.close()



def agent_overwiew(agent_path):
    
    agent = PPO.load(Path(agent_path))

    env_kwargs = {
        'continuous': CONTINUOUS, 
        'render_mode': "human"
    }
    eval_env = make_vec_env(ENVIRONMENT, n_envs=1, env_kwargs=env_kwargs)
    eval_env = VecFrameStack(eval_env, n_stack=4)

    terminated = [False]

    for episode in range(3):
        observation = eval_env.reset()
        while terminated[0] == False: 
            action, _states = agent.predict(observation, deterministic=True)
            observation, rewards, terminated, info = eval_env.step(action)
            
    eval_env.close()

if __name__ == "__main__":
    train_agent(static=True)
    # agent_overwiew(Path(ROOT_PATH, AGENT_DIR, AGENT_STATIC))