import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from pathlib import Path
import json

ROOT_PATH = Path(__file__).parent

with open(Path(ROOT_PATH, "config.json"), 'r') as f:
    config = json.load(f)
ENVIRONMENT = config["ENV"]["environment"]
RENDER_MODE = config["ENV"]["render_mode"]
CONTINUOUS = config["ENV"]["continuous"]
GRAVITY = config["ENV"]["gravity"]
ENABLE_WIND = config["ENV"]["enable_wind"]
WIND_POWER = config["ENV"]["wind_power"]
TURBULENCE_POWER = config["ENV"]["turbulence_power"]

NUM_EVAL_EPISODES = 3
RECORD_DIR = Path(ROOT_PATH, "records", "random")


def random_record_overview():

    ENV = gym.make(
        ENVIRONMENT,
        render_mode="rgb_array", 
        continuous=CONTINUOUS,
        gravity=GRAVITY,
        enable_wind=ENABLE_WIND,
        wind_power=WIND_POWER,
        turbulence_power=TURBULENCE_POWER
    )
    ENV = RecordVideo(ENV, video_folder=RECORD_DIR, name_prefix="eval", episode_trigger=lambda x: True)
    ENV = RecordEpisodeStatistics(ENV, buffer_length=NUM_EVAL_EPISODES)

    for _ in range(NUM_EVAL_EPISODES):  
        observation, info = ENV.reset()
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0

        while not terminated and not truncated:
            
            action = ENV.action_space.sample()
            # action[0]: steering (-1 to 1)
            # action[1]: acceleration (0 to 1)
            # action[2]: brake (0 to 1)
            
            observation, reward, terminated, truncated, info = ENV.step(action)
            
            total_reward += reward
            steps += 1


def random_interactive_overview():

    ENV = gym.make(
        ENVIRONMENT,
        render_mode="human", 
        continuous=CONTINUOUS,
        gravity=GRAVITY,
        enable_wind=ENABLE_WIND,
        wind_power=WIND_POWER,
        turbulence_power=TURBULENCE_POWER
    )

    for _ in range(NUM_EVAL_EPISODES):  
        observation, info = ENV.reset()
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0

        while not terminated and not truncated:
            
            action = ENV.action_space.sample()
            # action[0]: steering (-1 to 1)
            # action[1]: acceleration (0 to 1)
            # action[2]: brake (0 to 1)
            
            observation, reward, terminated, truncated, info = ENV.step(action)
            
            total_reward += reward
            steps += 1


if __name__ == "__main__":
    random_record_overview()