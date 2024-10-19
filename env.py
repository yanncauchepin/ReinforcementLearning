import gymnasium as gym

ENVIRONNEMENT = 'CarRacing-v2'
RENDER_MODE ='rgb_array'
LAP_COMPLETE_PERCENT = 0.95
DOMAIN_RANDOMIZE = False
CONTINOUS = False

ENV = gym.make(
    ENVIRONNEMENT, 
    render_mode=RENDER_MODE, 
    lap_complete_percent=LAP_COMPLETE_PERCENT, 
    domain_randomize=DOMAIN_RANDOMIZE, 
    continuous=CONTINOUS
    )