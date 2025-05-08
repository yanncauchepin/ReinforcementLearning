import gymnasium as gym

ENVIRONMENT = "CarRacing-v3"
RENDER_MODE = "rgb_array"
LAP_COMPLETE_PERCENT = 1.00
DOMAIN_RANDOMIZE = False
CONTINOUS = True

def random_overview():

    ENV = gym.make(
        ENVIRONMENT,
        render_mode="human", # For interactive rendering
        lap_complete_percent=LAP_COMPLETE_PERCENT,
        domain_randomize=DOMAIN_RANDOMIZE,
        continuous=CONTINOUS,
    )

    for _ in range(10):  
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
    random_overview()