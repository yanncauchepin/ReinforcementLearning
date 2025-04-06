from datetime import datetime

from env import ENV
from modules.dqn_agent import DQNAgent
from CarRacing.modules.test_env import (
    render_environment_agent,
    render_environment_random,
)


def train_agent():
    config = {
        "num_episodes": 5000,
        "max_steps": 2000,
        "alpha": 0.001,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_decay": 0.999,
        "min_epsilon": 0.1,
        "memory_size": 5000,
    }
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    dqn_agent = DQNAgent(f"rl_dqn_agent_{current_time}", **config)
    dqn_agent.train()


if __name__ == "__main__":
    train_agent()
    # ENV.close()
    # render_environment_agent("rl_dqn_agent_20241020002617")
