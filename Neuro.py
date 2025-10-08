# Main Script for Neuro Project 
# Version:  v 0.2.1
# Author:   K. E. Brown, Chad GPT.
# First:    2025-10-03
# Updated:  2025-10-06


# Imports
from audioop import avg
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
init_seed = 42069

# Agent Population
init_population = 15
agents = []

# Agent Initial Attribute Distribution
init_hunger = 30
init_thirst = 80
init_perception = 5
init_periferal = 3

# Neural Network Scrip
learning_rate = 0.1
epochs = 10000
sim_lifetime = 500

def main():
    # Loop Tracking
    champion_epoch = -1
    champion_lifespan = 0
    avg_lifespan = 0

    # Create Grid
    grid = Grid(width, height, init_seed)
    
    # Initialize Agents
    for _ in range(init_population):
       agents.append(Agent(0, 0, grid, init_perception, init_periferal, learning_rate, init_hunger, init_thirst))
                   
    # Training Loop
    for epoch in range(epochs):
        # Create Grid
        grid.reseed(init_seed + (epoch * epochs))
        
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        last_step = -1
        
        # Reset Grid
        grid.init()
        grid.render()

        # Reset Agents
        for agent in agents:
            agent.reset()

        grid.populate(agents)
        # for a in agents[:5]:
        #     print(a.x, a.y, grid.cells[a.y][a.x])
            
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
            if (step % (sim_lifetime / 5) == 0):
                grid.render()
                
            # Agents learn periodically
            if (step % (sim_lifetime / 100) == 0):
                for agent in agents:
                    if agent.memory:
                        agent.learn()


            last_step = step
        
        # Learn before end of epoch
        for agent in agents:
            if agent.memory:
                agent.learn()
                
        if (last_step > champion_lifespan):
            champion_lifespan = last_step
            champion_epoch = epoch
            print(f"New Champion Epoch: {champion_epoch+1} with Lifespan: {champion_lifespan}")
        else:
            print(f"Epoch Ended Below Champion: step {last_step} is {(champion_lifespan - last_step) * 100 / champion_lifespan:.2f}% below Champion: {champion_lifespan}.")
            
        if (epoch == 0):
            avg_lifespan = last_step
        else:
            avg_lifespan = (avg_lifespan + last_step) / 2
            
        print(f"Average Epoch Lifespan: {avg_lifespan:.2f}, This Run: {(last_step/avg_lifespan) * 100:.2f}% of avg")

        # Track Champion Epoch


        print(f"Epoch {epoch+1}/{epochs} completed.")
    
    # End of Sim Summary
    print(f"\nTraining completed. Champion Epoch: {champion_epoch+1} with Lifespan: {champion_lifespan}. Average Lifespan: {avg_lifespan}")

if __name__ == "__main__":
    main()