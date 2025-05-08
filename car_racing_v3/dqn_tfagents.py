import tensorflow as tf
import numpy as np
import gymnasium as gym
import json
from pathlib import Path
import os
import datetime

import tf_agents
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics
from tf_agents.drivers import dynamic_step_driver
from tf_agents.policies import random_tf_policy, policy_saver
from tf_agents.utils import common
from tf_agents.specs import BoundedArraySpec 

ROOT_PATH = Path(__file__).parent

with open(Path(ROOT_PATH, "config.json"), 'r') as f:
    config = json.load(f)
ENVIRONMENT = config["ENV"]["environment"]
LAP_COMPLETE_PERCENT = config["ENV"]["lap_complete_percent"]
DOMAIN_RANDOMIZE = config["ENV"]["domain_randomize"]
CONTINUOUS = config["DQN"]["continuous"]
MODEL_DIR = config["DQN"]["mdoel_dir"]
LOG_DIR = config["DQN"]["log_dir"]
CHECKPOINT_DIR = config["DQN"]["checkpoint_dir"]
POLICY_DIR = config["DQN"]["policy_dir"]

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
MODEL_NOW = f"tfagent_{now}"
MODEL_STATIC = "tfagent"



NUM_ITERATIONS = 10000  
COLLECT_STEPS_PER_ITERATION = 1
REPLAY_BUFFER_MAX_LENGTH = 100000
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
LOG_INTERVAL = 200
NUM_EVAL_EPISODES = 5
EVAL_INTERVAL = 1000
CONV_LAYER_PARAMS = ((32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1))
FC_LAYER_PARAMS = (512,)


os.makedirs(Path(ROOT_PATH, MODEL_DIR), exist_ok=True)
os.makedirs(Path(ROOT_PATH, LOG_DIR), exist_ok=True)
os.makedirs(Path(ROOT_PATH, CHECKPOINT_DIR), exist_ok=True)
os.makedirs(Path(ROOT_PATH, POLICY_DIR), exist_ok=True)


def train_agent(static=False):
    
    if static is False:
        model = Path(MODEL_NOW)
    else:
        model = Path(MODEL_STATIC)
    
    py_env = suite_gym.load(
        ENVIRONMENT, 
        gym_kwargs={
            'continuous': eval(CONTINUOUS), 
            'render_mode': None,
            'lap_complete_percent': LAP_COMPLETE_PERCENT,
            'domain_randomize': DOMAIN_RANDOMIZE
        }
    )
    tf_env = tf_py_environment.TFPyEnvironment(py_env)

    # For image observations, a QNetwork with convolutional layers is needed.
    q_net = q_network.QNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        conv_layer_params=CONV_LAYER_PARAMS,
        fc_layer_params=FC_LAYER_PARAMS
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    train_step_counter = tf.Variable(0)

    # DQN Agent
    # Note: TF-Agents DQN expects action_spec to be discrete.
    # If it's continuous, you might get errors or need a different agent (like DdpgAgent or SacAgent)
    # and potentially an action_spec_adapter if the default Box space has issues.
    agent = dqn_agent.DqnAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter
    )
    agent.initialize()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=REPLAY_BUFFER_MAX_LENGTH
    )

    eval_policy = agent.policy
    collect_policy = agent.collect_policy
    random_policy = random_tf_policy.RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec())

    replay_observer = [replay_buffer.add_batch]
    collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        collect_policy,
        observers=replay_observer,
        num_steps=COLLECT_STEPS_PER_ITERATION
    )

    collect_driver.run = common.function(collect_driver.run)
    agent.train = common.function(agent.train)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=BATCH_SIZE,
        num_steps=2 # For N-step Q-learning, num_steps=N+1
    ).prefetch(3)
    iterator = iter(dataset)


    initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        random_policy,
        observers=replay_observer,
        num_steps=1000
    )
    initial_collect_driver.run()

    time_step = tf_env.reset()
    for _ in range(NUM_ITERATIONS):
        time_step, _ = collect_driver.run(time_step) 
        experience, _ = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % LOG_INTERVAL == 0:
            print(f'step = {step}, loss = {train_loss}')

        if step % EVAL_INTERVAL == 0:
            avg_return = compute_avg_return(
                tf_py_environment.TFPyEnvironment(
                    suite_gym.load(
                        ENVIRONMENT, 
                        gym_kwargs={
                            'continuous': eval(CONTINUOUS), 
                            'render_mode':None,
                            'lap_complete_percent': LAP_COMPLETE_PERCENT,
                            'domain_randomize': DOMAIN_RANDOMIZE
                        }
                    )
                ),
                eval_policy, NUM_EVAL_EPISODES
            )
            print(f'step = {step}, Average Return = {avg_return}')
    
    tf_policy_saver = policy_saver.PolicySaver(agent.policy)
    tf_policy_saver.save(Path(ROOT_PATH, POLICY_DIR, model))

    py_env.close()


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def agent_overwiew(model_path):

    saved_policy = tf.saved_model.load(Path(model_path))


    eval_py_env = suite_gym.load(ENVIRONMENT, gym_kwargs={'continuous': eval(CONTINUOUS), 'render_mode': "human"})

    for i in range(3):
        time_step = eval_py_env.reset()
        eval_py_env.render() 
        episode_return = 0
        steps = 0
        while not time_step.is_last():
            
            batched_time_step = tf_agents.trajectories.time_step.restart(
                observation=tf.expand_dims(tf.constant(time_step.observation, dtype=tf.uint8), axis=0), 
                reward=tf.expand_dims(tf.constant(time_step.reward, dtype=tf.float32), axis=0),
                discount=tf.expand_dims(tf.constant(time_step.discount, dtype=tf.float32), axis=0),
                step_type=tf.expand_dims(tf.constant(time_step.step_type, dtype=tf.int32), axis=0)
            )
            action_step = saved_policy.action(batched_time_step)
            time_step = eval_py_env.step(action_step.action.numpy()[0]) 
            episode_return += time_step.reward
            steps +=1
            # eval_py_env.render() # Render is automatic with gym.make(render_mode="human") on step

    eval_py_env.close()


if __name__ == "__main__":
    train_agent(static=True)
    agent_overwiew(Path(ROOT_PATH, POLICY_DIR, MODEL_STATIC))