# Agent Class, a.k.a. "Lil' Guys"
# Author:   K. E. Brown, Chad GPT.
# First:    2025-10-03
# Updated:  2025-10-06

# Imports
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from network import ActorCriticNetwork  # assuming your network script is 'network.py'

# Import shared symbols and grid reference
from grid import WATER, FOOD, DANGER, EMPTY, AGENT, CORPSE

class Agent:
    # Directions (8-way movement)
    DIRECTIONS = [
        (0, -1),   # N
        (1, -1),   # NE
        (1, 0),    # E
        (1, 1),    # SE
        (0, 1),    # S
        (-1, 1),   # SW
        (-1, 0),   # W
        (-1, -1)   # NW
    ]

    def __init__(self, x, y, grid, sight_range=3, cone_width=3, learning_rate=.002, base_hunger=15, base_thirst=45):
        self.x = x
        self.y = y
        self.dir = 0  # facing north initially
        self.sight_range = sight_range
        self.cone_width = max(1, min(8, cone_width))
        self.grid = grid
        self.alive = True
        self.symbol = AGENT
        
        # Hunger / Thirst
        self.hunger_max = base_hunger
        self.thirst_max = base_thirst
        self.hunger = self.hunger_max
        self.thirst = self.thirst_max
        
        # input_size now guaranteed to match perceive() output
        input_size = self.sight_range * self.cone_width
        hidden_size = 16
        action_size = 8
        self.network = ActorCriticNetwork(input_size, hidden_size, action_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        
        # Brain Stuff
        self.memory = []
        
    # --- Perception ---
    def perceive(self):
        """
        Raymarch outward from the agent within the cone of vision.
        Returns a 1D array of numeric perception values.
        """
        perception = []

        # Determine which directions fall in the cone
        half_cone = self.cone_width // 2
        directions = [(self.dir + i) % 8 for i in range(-half_cone, half_cone + 1)]

        for d in directions:
            dx, dy = self.DIRECTIONS[d]
            ray_values = []
            for r in range(1, self.sight_range + 1):
                tx = self.x + dx * r
                ty = self.y + dy * r
                # Check bounds
                if tx < 0 or ty < 0 or ty >= len(self.grid.cells) or tx >= len(self.grid.cells[0]):
                    break  # outside grid

                cell = self.grid.cells[ty][tx]
            
                
                if cell == DANGER:
                    ray_values.append(-999)
                    break
                elif cell == CORPSE:
                    ray_values.append(-20)
                elif cell == WATER:
                    ray_values.append(1)
                elif cell == FOOD:
                    ray_values.append(20)
                elif cell == AGENT:
                    ray_values.append(-1)
                else:
                    ray_values.append(0)

            # pad ray to sight_range length
            while len(ray_values) < self.sight_range:
                ray_values.append(0)

            perception.extend(ray_values)
            
        while len(perception) < self.sight_range * self.cone_width:
            perception.append(0)


        # Normalize for neural net input
        return np.array(perception, dtype=np.float32)
    
    # --- Decision Making ---
    def decide(self, obs):
        # Get policy and value from the network
        policy, value = self.network(obs)

        # Convert to numpy and flatten
        policy = policy.numpy()[0]

        # Normalize just in case of rounding errors
        policy = policy / np.sum(policy)

        # Choose action based on probabilities
        action = np.random.choice(len(policy), p=policy)

        return action, value

    # --- Movement ---
    def move(self, action):
        reward = 0
        # Death check
        if (self.is_dead()):
            self.grid.mark_corpse(self.x, self.y)
            reward = -20
            return reward   
            
        self.dir = action
        dx, dy = self.DIRECTIONS[action]
        nx, ny = self.x + dx, self.y + dy

        if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
            cell = self.grid.cells[ny][nx]

            # --- Eat / Drink logic ---
            if cell == FOOD:
                reward = 50/self.hunger
                self.hunger = self.hunger_max
                cell = EMPTY  # consume food
            elif cell == WATER:
                reward = 20/self.thirst
                self.thirst = self.thirst_max

            # Update grid positions
            self.grid.cells[self.y][self.x] = EMPTY
            self.x, self.y = nx, ny
            self.grid.cells[self.y][self.x] = AGENT

        # check bounds
        if 0 <= nx < len(self.grid.cells[0]) and 0 <= ny < len(self.grid.cells):
            self.x, self.y = nx, ny
            
        # --- Update ---
        self.hunger -= 1
        self.thirst -= 1        

        return reward
    
    # --- Education ---
    def learn(self, gamma=0.99):
        for obs, action, reward, next_obs, alive in self.memory:
            done = 0.0 if alive else 1.0
            obs = tf.convert_to_tensor([obs], dtype=tf.float32)
            next_obs = tf.convert_to_tensor([next_obs], dtype=tf.float32)

        # Compute target and advantage
        _, value = self.network(obs)
        _, next_value = self.network(next_obs)
        target = reward + (1 - done) * gamma * next_value
        delta = target - value

        with tf.GradientTape() as tape:
            policy, value = self.network(obs)
            action_prob = policy[0, action]
            actor_loss = -tf.math.log(tf.cast(action_prob, tf.float32) + 1e-8) * delta
            critic_loss = tf.square(delta)
            total_loss = actor_loss + critic_loss

        grads = tape.gradient(total_loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))

        # Clear memory after learning
        self.memory = []
    
    def is_dead(self, force=False):
        self.alive = not (force or self.hunger <= 0 or self.thirst <= 0)
        return not self.alive
    
    # --- Reset ---
    def reset(self):
        self.hunger = self.hunger_max
        self.thirst = self.thirst_max
        self.alive = True
        self.grid.cells[self.y][self.x] = AGENT
