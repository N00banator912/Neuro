# Agent Class, a.k.a. "Lil' Guys"
# Author:   K. E. Brown, Chad GPT.
# First:    2025-10-03
# Updated:  2025-10-11

# Imports
import os
from keras import activations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from network import ActorCriticNetwork  # assuming your network script is 'network.py'
import random
import matplotlib
import matplotlib.pyplot as plt

# Import shared symbols and grid reference
from grid import WATER, FOOD, DANGER, EMPTY, AGENT, CORPSE

matplotlib.use("Agg")  # Non-interactive, no GUI required

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
        self.dir = random.randint(0, 7)  # Random initial direction
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
        self.hunger_loss = 1
        self.thirst_max = base_thirst
        self.thirst_loss = 1
        self.hunger = self.hunger_max
        self.thirst = self.thirst_max
        self.health = self.health_max
        self.power = 1
        self.age = 0
        
        # Achievement Attributes
        self.times_eaten = 0
        self.times_drank = 0
        self.steps_in_pain = 0
        self.was_in_pain = False
        self.happiness_max = 1.0        # Maximum Happiness ever recorded
        self.happiness_min = 1.0        # Minimum Happiness ever recorded
        self.happiness_total = 0.0      # Total Lifetime Happiness
        self.death_step = None          # Deathdate
        
        # Movement Attributes
        self.visited = set()
        self.last_action = None
        self.last_failed = False
        self.tile_under = EMPTY

        
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
                ray_values.append(-5)

            perception.extend(ray_values)
            
        while len(perception) < self.sight_range * self.cone_width:
            perception.append(-5)


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

        # Decrease probability of repeating last failed action
        if self.last_failed and self.last_action is not None:
            policy[self.last_action] *= 0.25  # reduce its weight drastically
            policy = policy / np.sum(policy)  # renormalize
        elif self.last_action == 8:  # if last action was 'sit'
            policy[self.last_action] *= 0.66  # reduce its weight moderately
            policy = policy / np.sum(policy)  # renormalize


        # Choose action based on probabilities
        action = np.random.choice(len(policy), p=policy)

        return action, value

    # --- Movement ---
    def move(self, action):
        event = None
        if self.is_dead():
            self.grid.mark_corpse(self.x, self.y)
            return self.compute_reward(event="death")

        self.dir = (self.dir + action) % len(self.DIRECTIONS)

            # Handle movement / sit still
        if action >= len(self.DIRECTIONS) - 1:  # last action is sit still
            dx, dy = 0, 0
            self.happiness *= 0.9  # Slight decrease for being idle
            event = "idle"
        else:
            dx, dy = self.DIRECTIONS[action]
         
            
        nx, ny = self.x + dx, self.y + dy

        # Bounds Checking
        if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
            cell = self.grid.cells[ny][nx]
            if cell == FOOD:
                self.hunger = self.hunger_max
                self.happiness *= 1.1
                event = "eat"
                self.grid.cells[ny][nx] = EMPTY
                self.times_eaten += 1
            elif cell == CORPSE:
                self.hunger = self.hunger_max * .5
                self.happiness *= .5
                event = "eat"
                self.grid.cells[ny][nx] = EMPTY
                self.times_eaten += 1
            elif cell == WATER:
                self.thirst = self.thirst_max
                self.happiness *= 1.05
                event = "drink"
                self.times_drank += 1
            elif cell == DANGER:
                self.hurt(3)
                event = "danger"
            elif cell == AGENT:
                # Engage in combat
                target = self.grid.get_agent_at(nx, ny)  # You'll need to add this helper in Grid (shown below)
                if target and target.alive:
                    event = "fight"
                    self.fight(target)
                else:
                    # If no valid target found (shouldn’t happen)
                    event = "bump"
                nx, ny = self.x, self.y  # Stay in place after fighting

            # All other cells Impassable
            elif cell != EMPTY:
                nx, ny = self.x, self.y
                self.happiness *= 0.9  # Slight decrease for failed move
                event = "bump"      
            else:
                self.happiness *= 1.1  # Slight increase for successful move
                event = "move"

            if (nx, ny) not in self.visited:
                self.visited.add((nx, ny))
                self.happiness *= 1.5  # Slight increase for exploring

            # Move agent
            self.grid.cells[self.y][self.x] = self.tile_under
            self.tile_under = self.grid.cells[ny][nx]
            if self.tile_under == FOOD:
                self.tile_under = EMPTY  # Food will be eaten when stepped on, so we don't want it to be replaced when moving away
            self.x, self.y = nx, ny
            self.grid.cells[self.y][self.x] = AGENT
        else:   # Out of bounds
            self.happiness *= 0.99  # Slight decrease for failed move
            event = "bump"
            
        self.last_failed = (event == "bump")
        self.last_action = action

            
        # Tick down hunger, thirst, etc.
        self.biostasis()
        
        #if event in ["eat"]:
        #    print(f"Agent {id(self)} ate(x={self.x}, y={self.y}), Hunger: {self.hunger}, Happiness: {self.happiness:.2f}")
        #elif event in ["drink"]:
        #    print(f"Agent {id(self)} drank(x={self.x}, y={self.y}), Thirst: {self.thirst}, Happiness: {self.happiness:.2f}")
            

        if self.is_dead():
            self.grid.mark_corpse(self.x, self.y)
            event = "death"
            print(f"💀 Agent {id(self)} died at age {self.age} position({self.x}, {self.y})")

        return self.compute_reward(event)
    
    # --- Act ---
    def act(self, shared_network):
        obs = self.perceive()
        policy, value = shared_network(tf.convert_to_tensor([obs], dtype=tf.float32))
        action = np.random.choice(len(policy[0]), p=policy[0].numpy())
        return obs, action, value
        
    # --- Biostasis (a.k.a. Health Update) ---
    def biostasis(self):
        self.age += 1
        self.hunger -= self.hunger_loss
        self.thirst -= self.thirst_loss

        # secondary health effects
        if self.hunger <= 0:
            self.hurt(1)
        elif self.hunger >= self.hunger_max and self.health < self.health_max:
            self.hurt(-1)
        
        if self.thirst <= 0:
            self.hurt(1)
        elif self.thirst >= self.thirst_max and self.health < self.health_max:
            self.hurt(-1)
            
        if self.happiness <= .6:
            self.hurt(1)
        if self.happiness >= .9 and self.health < self.health_max:
            self.hurt(-1)
            
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
        reward = 0.01

        # Strong event-based signals
        if event == "eat":
            reward += 1.00
        elif event == "drink":
            reward += 0.50
        elif event == "death":
            reward -= 1.00
        elif event == "danger":
            reward -= 0.50
        elif event == "idle":
            reward -= 0.20
        elif event == "bump":
            reward -= 0.05
        elif event == "move":
            reward += 0.02
        elif event == "fight":
            if self.is_dead():
                reward -= 1.0
                print(f"💀 Agent {id(self)} died in battle at age {self.age} position({self.x}, {self.y})")
            else:
            # Encourage victorious combat, discourage wasteful fights
                reward += 0.3 if self.happiness > 1.0 else -0.3

            

        # Penalize low hunger/thirst gradually
        hunger_penalty = max(0, 1 - (self.hunger / max(1, self.hunger_max)))
        thirst_penalty = max(0, 1 - (self.thirst / max(1, self.thirst_max)))
        reward -= 0.5 * (hunger_penalty + thirst_penalty)

        # Small health-based adjustment
        health_frac = self.health / max(1.0, self.health_max)
        reward += (health_frac - 1.0) * 0.2  # small tweak: negative when hurt

        # Decrease if in Pain
        if self.health < self.health_max * self.pain_threshold:
            reward *= 0.75
            
        # Multiply by happiness factor
        self.happiness = np.clip(self.happiness, 0.5, 1.5)
        reward *= self.happiness

        # Keep reward within sane range
        # print (f"Agent {id(self)} Reward (pre-clip): {reward}")
        reward = np.clip(reward, -1.0, 1.0)
        # print (f"Agent {id(self)}, Action: {event}, Reward: {reward:.1f}, Happiness: {self.happiness:.3f}")
        return reward

    # --- Policy Visualization ---
    def visualize_policy_grid(self, shared_network, logger=None, step=None):
        """
        Visualize the agent's current policy as a 3x3 grid of action probabilities.
        The grid maps directional actions to their spatial equivalents.
        """
        obs = self.perceive()
        policy, _ = shared_network(tf.convert_to_tensor([obs], dtype=tf.float32))
        policy = policy.numpy()[0]
        policy = policy / np.sum(policy)

        # 3x3 layout mapping actions to direction
        grid_layout = np.array([
            [policy[7], policy[0], policy[1]],  # NW, N, NE
            [policy[6], policy[8], policy[2]],  # W, SIT, E
            [policy[5], policy[4], policy[3]]   # SW, S, SE
        ])

        fig, ax = plt.subplots(figsize=(3, 3))
        im = ax.imshow(grid_layout, cmap="plasma", interpolation="nearest")

        # Add probability text
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{grid_layout[i, j]:.2f}",
                        ha="center", va="center", color="white", fontsize=10)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Policy Grid — Agent at ({self.x}, {self.y})")

        # Optional colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Action Probability")

        if logger:
            logger.log_figure("policy_grid", fig, step=step)
        plt.close(fig)


        
    # --- Hurt Function ---
    def hurt(self, damage):
        self.health -= damage
        if self.health < 0:
            self.health = 0
            self.is_dead(force=True)
        elif self.health > self.health_max:
            self.health_max += 1
            self.health = self.health_max
        return self.health

    def fight(self, target):
        """
        Simple combat resolution between two agents.
        Returns True if this agent wins, False otherwise.
        """
        if not target.alive:
            return True  # target already dead
        # Base combat logic: higher happiness & health gives advantage
        self_power = self.health + self.happiness * random.uniform(0.8, 1.2)
        target_power = target.health + target.happiness * random.uniform(0.8, 1.2)
    
        if hasattr(self, "grid"):
            self.grid.fight_heatmap[self.y, self.x] += 1.0
        
        if self_power >= pow(target_power, 2):    
            # Insta kill
            target.is_dead(force=True)
            target.happiness *= 0.2
            self.happiness *= 1.05  # morale boost
        elif self_power >= target_power * 2.0:
            # Attacker wins
            target.hurt(self.power * 2.0)
            target.happiness *= 0.4
            self.happiness *= 0.99  # combat stress
            return True
        elif self_power >= target_power:
            # Attacker wins
            target.hurt(self.power)
            target.happiness *= 0.8
            self.happiness *= 0.95  # combat stress
            self.hurt(0.5)  # minor injury
            return True        
        elif self_power < target_power:
            # Defender wins
            self.hurt(self.power)
            self.happiness *= 0.8
            target.happiness *= 0.95  # stress for both
            return False        
        else:
            # Tie: both take damage
            self.hurt(self.power * 0.5)
            target.hurt(self.power * 0.5)
            self.happiness *= 0.9
            target.happiness *= 0.9
            return None


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
        self.visited.clear()

        # Place back on the grid
        self.grid.cells[self.y][self.x] = AGENT

    # Helper Functions
    # Set Trainer
    def set_trainer(self, trainer):
        """
        Assign a centralized trainer to the agent.
        """
        self.trainer = trainer
