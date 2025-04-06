import pygame
import tensorflow as tf
import numpy as np

from env import ENV

pygame.init()


def render_environment_random():
    # Create a display window
    screen = pygame.display.set_mode((800, 800))
    pygame.display.set_caption("Random Agent Visualization")

    for _ in range(10):  # Run 10 episodes for visualization
        state, _ = ENV.reset()
        done = False

        while not done:
            action = ENV.action_space.sample()  # Random action for demonstration
            state, reward, done, _, _ = ENV.step(action)

            # Convert the state from uint8 to a format suitable for pygame
            state_surface = pygame.surfarray.make_surface(state)
            state_surface = pygame.transform.scale(
                state_surface, (800, 800)
            )  # Resize for better visibility

            # Display the image
            screen.blit(state_surface, (0, 0))
            pygame.display.flip()

            # Event handling to quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # pygame.time.delay(0)  # Control the speed of rendering

    pygame.quit()


def render_environment_agent(artefact_agent):
    agent = tf.keras.models.load_model(f"trained_agents/{artefact_agent}.keras")

    # Create a display window
    screen = pygame.display.set_mode((800, 800))
    pygame.display.set_caption("Agent Visualization")

    for _ in range(10):  # Run 10 episodes for visualization
        state, _ = ENV.reset()
        state = np.reshape(state, [1, 96, 96, 3])  # Reshape for the model input
        done = False

        while not done:
            action = np.argmax(
                agent.predict(state)[0]
            )  # Use the model to predict action
            next_state, reward, done, _, _ = ENV.step(action)
            state = np.reshape(
                next_state, [1, 96, 96, 3]
            )  # Prepare the next state for prediction

            # Convert the state from uint8 to a format suitable for pygame
            state_surface = pygame.surfarray.make_surface(next_state)
            state_surface = pygame.transform.scale(
                state_surface, (800, 800)
            )  # Resize for better visibility

            # Display the image
            screen.blit(state_surface, (0, 0))
            pygame.display.flip()

            # Event handling to quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # pygame.time.delay(0)  # Control the speed of rendering

    pygame.quit()
