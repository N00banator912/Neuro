# Main Script for Neuro Project 
# Version:  v 0.2.1
# Author:   K. E. Brown, Chad GPT.
# First:    2025-10-03
# Updated:  2025-10-06


# Imports
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

from tkinter import Grid
import numpy as np
from grid import Grid, AGENT, EMPTY
from agent import Agent

# Grid Script
width = 50
height = 20

# Agent Population
init_population = 15
agents = []

# Agent Initial Attribute Distribution
init_hunger = 30
init_thirst = 80
init_perception = 5
init_periferal = 3

# Loop Tracking
champion_epoch = -1
champion_lifespan = 0

# Neural Network Scrip
learning_rate = 0.1
epochs = 10000
sim_lifetime = 500

def main():
    # Initialize Grid
    grid = Grid(width, height)
    
    # Initialize Agents
    for _ in range(init_population):
       agents.append(Agent(0, 0, grid, init_perception, init_periferal, learning_rate, init_hunger, init_thirst))
                   
    # Training Loop
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        last_step = -1
        
        # Reset Grid
        grid.init()
        grid.render()

        # Reset Agents
        for agent in agents:
            if (agent.is_dead()):
                agent.reset()

        grid.populate(agents)
        
        # Simulation Loop
        for step in range(sim_lifetime):
            alive_agents = [a for a in agents if a.alive]
            if not alive_agents:
                print(f"All agents dead at step {step}. Starting next epoch.")
                break  # End this epoch early
        
            for agent in agents:
                if not agent.alive:
                    continue
                obs = agent.perceive()
                action, value = agent.decide(tf.convert_to_tensor([obs], dtype=tf.float32))
                
                reward = agent.move(action)
                next_obs = agent.perceive()
                
                agent.memory.append((obs, action, reward, next_obs, agent.alive))
            
            # Spawn new food periodically
            if (step % (sim_lifetime / 10) == 0):
                grid.spawn_food()
            
            # Render grid periodically
            if (step % (sim_lifetime / 50) == 0):
                grid.render()

            last_step = step
        
        # Learn before end of epoch
        for agent in agents:
            if agent.memory:
                agent.learn()
                
        if (last_step > self.champion_lifespan):
            self.champion_lifespan = last_step
            self.champion_epoch = epoch
            print(f"New Champion Epoch: {champion_epoch+1} with Lifespan: {champion_lifespan}")
        else:
            print(f"Epoch Ended Below Champion: step {last_step} is {(self.champion_lifespan - last_step) / self.champion_lifespan}% below champion.")

        # Track Champion Epoch


        print(f"Epoch {epoch+1}/{epochs} completed.")

if __name__ == "__main__":
    main()