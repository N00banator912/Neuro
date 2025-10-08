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
        (-1, -1),  # NW
        (0, 0)     # Sit
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

        # Health Attributes
        self.happiness = 1.0
        self.health_max = 10
        self.damage_threshold = 1
        self.pain_threshold = .65
        self.hunger_max = base_hunger
        self.thirst_max = base_thirst
        self.hunger = self.hunger_max
        self.thirst = self.thirst_max
        self.health = self.health_max
        
        # Achievement Attributes
        self.times_eaten = 0
        self.times_drank = 0
        self.steps_in_pain = 0
        self.was_in_pain = False
        self.happiness_max = 1.0        # Maximum Happiness ever recorded
        self.happiness_min = 1.0        # Minimum Happiness ever recorded
        self.happiness_total = 0.0      # Total Lifetime Happiness
        self.death_step = None          # Deathdate
        
        # input_size now guaranteed to match perceive() output
        input_size = self.sight_range * self.cone_width
        hidden_size = 32
        action_size = 8
        
        # Brain Stuff
        self.local_memory = []
        
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
        policy, value = self.trainer.network(obs)


        # Convert to numpy and flatten
        policy = policy.numpy()[0]

        # Normalize just in case of rounding errors
        policy = policy / np.sum(policy)

        # Choose action based on probabilities
        action = np.random.choice(len(policy), p=policy)

        return action, value

    # --- Movement ---
    def move(self, action):
        event = None
        if self.is_dead():
            self.grid.mark_corpse(self.x, self.y)
            return self.compute_reward(event="death")

        self.dir = action

            # Handle movement / sit still
        if action >= len(self.DIRECTIONS) - 1:  # last action is sit still
            dx, dy = 0, 0
            self.happiness *= 0.98  # Slight decrease for being idle
            event = "idle"
        else:
            dx, dy = self.DIRECTIONS[action]
         
            
        nx, ny = self.x + dx, self.y + dy

        if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
            cell = self.grid.cells[ny][nx]
            if cell == FOOD:
                self.hunger = self.hunger_max
                self.happiness *= 1.1
                event = "eat"
                self.grid.cells[ny][nx] = EMPTY
                self.times_eaten += 1
            elif cell == CORPSE:
                self.hunger = self.hunger_max
                self.happiness *= .5
                event = "eat"
                self.grid.cells[ny][nx] = EMPTY
                self.times_eaten += 1
            elif cell == WATER:
                self.thirst = self.thirst_max
                self.happiness *= 1.05
                event = "drink"
                self.grid.cells[ny][nx] = EMPTY
                self.times_drank += 1
            elif cell == DANGER:
                self.hurt(3)
                event = "danger"

            # Move agent
            self.grid.cells[self.y][self.x] = EMPTY
            self.x, self.y = nx, ny
            self.grid.cells[self.y][self.x] = AGENT

        # Tick down hunger and thirst
        self.hunger -= 1
        self.thirst -= 1
        self.biostasis()
            
        if self.is_dead():
            self.grid.mark_corpse(self.x, self.y)
            event = "death"

        return self.compute_reward(event)
    
    # --- Act ---
    def act(self, shared_network):
        obs = self.perceive()
        policy, value = shared_network(tf.convert_to_tensor([obs], dtype=tf.float32))
        action = np.random.choice(len(policy[0]), p=policy[0].numpy())
        return obs, action, value

    
    # --- Biostasis (a.k.a. Health Update) ---
    def biostasis(self):
        if self.hunger <= 0:
            self.hurt(1)
        if self.thirst <= 0:
            self.hurt(1)
            
        if self.hunger > self.hunger_max:
            self.hunger_max += 5
            self.hunger = self.hunger_max
        if self.thirst > self.thirst_max:
            self.thirst_max += 5
            self.thirst = self.thirst_max
            
        if self.health < self.health_max * self.pain_threshold:
            self.happiness *= .95
    
    # --- Compute Reward ---
    def compute_reward(self, event=None):
        """
        Compute the reward for the agent.
        event: optional string to emphasize specific event triggers like 'eat', 'drink', or 'death'.
        """
        # Base survival reward — just staying alive
        reward = 0.1

        # Strong event-based signals
        if event == "eat":
            reward += 10.0
        elif event == "drink":
            reward += 5.0
        elif event == "death":
            reward -= 20.0
        elif event == "danger":
            reward -= 10.0
        elif event == "idle":
            reward -= 5.0
            

        # Penalize low hunger/thirst gradually
        hunger_penalty = max(0, 1 - (self.hunger / self.hunger_max))
        thirst_penalty = max(0, 1 - (self.thirst / self.thirst_max))
        reward -= 2.0 * (hunger_penalty + thirst_penalty)

        # Small health-based adjustment
        reward += (self.health / self.health_max - 1) * 5.0

        # Decrease if in Pain
        if self.health < self.health_max * self.pain_threshold:
            reward *= 0.75
        # Multiply by happiness factor
        reward *= self.happiness

        # Keep reward within sane range
        reward = np.clip(reward, -25.0, 15.0)
        return reward

        
    # --- Hurt Function ---
    def hurt(self, damage):
        self.health -= damage
        if self.health < 0:
            self.health = 0
            self.is_dead(force=True)
        return self.health

    # --- Death Check ---
    def is_dead(self, force=False):
        self.alive = not (force or self.health <= 0)
        return not self.alive
    
    def reset(self):
        # Remove agent’s old position if needed
        if self.grid.cells[self.y][self.x] == AGENT:
            self.grid.cells[self.y][self.x] = EMPTY

        # Restore stats
        self.hunger = self.hunger_max
        self.thirst = self.thirst_max
        self.health = self.health_max
        self.happiness = 1.0
        self.alive = True
        self.times_eaten = 0
        self.times_drank = 0
        self.steps_in_pain = 0
        self.was_in_pain = False
        self.death_step = None

        # Reset memory and temporary learning buffers
        self.local_memory.clear()

        # Place back on the grid
        self.grid.cells[self.y][self.x] = AGENT

    def set_trainer(self, trainer):
        """
        Assign a centralized trainer to the agent.
        """
        self.trainer = trainer
