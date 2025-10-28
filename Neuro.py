# Main Script for Neuro Project 
# Version:  v 0.2.1
# Author:   K. E. Brown, Chad GPT.
# First:    2025-10-03
# Updated:  2025-10-26


# Imports
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

import random
from tkinter import Grid
import numpy as np


# Project Imports
from grid import Grid
from agent import Agent
from training import Trainer
from stats import StatsTracker
from logger import Logger

# Grid Script
width = 50
height = 20
init_seed = 42069

# Agent Population
init_population = 20
agents = []

# Agent Initial Attribute Distribution
init_hunger = 25
init_thirst = 50

# Network/Trainer Script
learning_rate = 0.1
hidden_size = 32
action_size = 9     # 8 directions + stay put

# writer = SummaryWriter(log_dir="runs/exp1")
# writer.add_scalar("debug/test_value", 1.0, 0)
# writer.flush()

# Local Time Variables
epochs = 1000
sim_lifetime = 500
learn_timer = 2
prnt_timer = 3
network_print_timer = 5

# Food Spawning Variables
food_timer = 10
food_count = 5

# Names Path
names_path = "SillyNames.txt"

def main():
    # Loop Tracking
    champion_epoch = -1
    champion_steps = 0
    champion_agent = None
    champion_lifespan = 0
    avg_lifespan = 0

    # Create the Stat Tracker and Logger
    stats = StatsTracker(total_epochs=epochs)
    logger = Logger(log_dir="logs/neuro")
 
    # Create Grid
    grid = Grid(width, height, init_seed)
    
    # Create the Trainer
    trainer = Trainer(input_size=, hidden_size=hidden_size, action_size=action_size, lr=learning_rate)

    # Open Name File
    with open("SillyNames.txt", "r", encoding="utf-8") as f:
        ALL_NAMES = [line.strip() for line in f if line.strip()]
    
    USED_NAMES = set()

    # Pull a name from the 
    def get_random_name():
        available = [n for n in ALL_NAMES if n not in USED_NAMES]
        if not available:
            # fallback if we run out
            name = f"Jeff {random.randint(1, 12538242069)}"
        else:
            name = random.choice(available)
            USED_NAMES.add(name)
        return name

    # Initialize Agents
    for _ in range(init_population):
        a = Agent(grid._rand_pos(x,y), grid, init_perception, init_periferal, learning_rate, init_hunger, init_thirst, get_random_name())
        a.set_trainer(trainer)
        agents.append(a)       

    # Draw initial policy grid
    if logger:
        agent = random.choice(agents)
        agent.visualize_policy_grid(trainer.network, logger=logger, step=0)            

    # --- Training Loop ---
    for epoch in range(epochs):

        grid.reseed((init_seed + (epoch * epochs)) * np.e)
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")

        # Reset grid and agents
        grid.init()
        for agent in agents:
            agent.reset()
        grid.populate(agents)

        last_step = -1

        # Simulation Loop
        for step in range(sim_lifetime):
            alive_agents = [a for a in agents if a.alive]
            dead_agents = [a for a in agents if not a.alive]
            if not alive_agents:
                grid.render()
                print(f"All agents dead at step {step}. Starting next epoch.")
                break

            # --- Agent Actions ---
            for agent in agents:
                if not agent.alive:
                    # Give one final strong negative transition on death
                    if hasattr(agent, "last_obs") and hasattr(agent, "last_action"):
                        trainer.store(agent.last_obs, agent.last_action, -20.0, agent.last_obs, False)
                    continue
                
                # Perceive environment
                obs = agent.perceive()
                agent.last_obs = obs
                obs_tensor = tf.convert_to_tensor([obs], dtype=tf.float32)
                
                # Get action + predicted observation (for potential advantage estimation)
                # Print the first and last decide calls
                # if agent == agents[0] or agent == agents[-1]:
                #    action, value = agent.decide(obs_tensor, debug=True)
                #else:
                action, value = agent.decide(obs_tensor)
                agent.last_action = action
                
                # Execute action
                reward = agent.move(action)
                next_obs = agent.perceive()

                # Store transition
                trainer.store(obs, action, reward, next_obs, agent.alive)

            # --- Environment Maintenance ---
            if step % food_timer == 0:
                grid.spawn_food(food_count)

            # Render periodically for debugging
            if step % prnt_timer == 0:
                grid.render()

            # --- Periodic Learning ---
            if step % learn_timer == 0 and trainer.memory:
                trainer.learn()

            last_step = step

        # --- End-of-Epoch Learning ---
        if trainer.memory:
            trainer.learn()
        
        # --- Print Network Logger ---
        if epoch % network_print_timer == 0:
            agent = random.choice(agents)
            agent.visualize_policy_grid(trainer.network, logger=logger, step=epoch)            

        # --- Stats Tracking ---
        if last_step > champion_lifespan:
            champion_lifespan = last_step
            champion_epoch = epoch
            print(f"🏆 New Champion Epoch: {champion_epoch+1} with Lifespan: {champion_lifespan}")
        else:
            diff = (champion_lifespan - last_step) * 100 / champion_lifespan
            print(f"Epoch Ended Below Champion: step {last_step} is {diff:.2f}% below Champion: {champion_lifespan}.")

        avg_lifespan = last_step if epoch == 0 else (avg_lifespan + last_step) / 2
        print(f"Average Epoch Lifespan: {avg_lifespan:.2f}, This Run: {(last_step / avg_lifespan) * 100:.2f}% of avg")

        print(f"Epoch {epoch+1}/{epochs} completed.")
        
        # --- Re-alive Agents for Next Epoch ---
        for agent in agents:
            agent.reset()

    # --- Summary ---
    print(f"\nTraining completed. Champion Epoch: {champion_epoch+1} with Lifespan: {champion_lifespan}.")
    print(f"Average Lifespan: {avg_lifespan:.2f}")
    
    # End of Sim Summary
    print(f"\nTraining completed. Champion Epoch: {champion_epoch+1} with Lifespan: {champion_lifespan}. Average Lifespan: {avg_lifespan}")

if __name__ == "__main__":
    main()