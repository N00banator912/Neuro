# Agent Class, a.k.a. "Lil' Guys"
# Author:   K. E. Brown, Chad GPT.
# First:    2025-10-03
# Updated:  2025-10-26

# Imports
from math import floor
import os
from keras import activations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from network import ActorCriticNetwork  # assuming your network script is 'network.py'
import random
import matplotlib
import matplotlib.pyplot as plt

# Local Imports
from grid import GRAVE, WATER, FOOD, DANGER, EMPTY, AGENT, CORPSE
from food import FLAVOR, compatibility

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

    def __init__(self, x, y, grid, perception=(2, 7), learning_rate=.002, base_hunger=15, base_thirst=45, name="Jeff", flavor=FLAVOR[0]):
        self.x = x
        self.y = y
        self.dir = random.randint(0, 7)  # Random initial direction
        self.grid = grid
        self.alive = True
        self.symbol = AGENT
        self.name = name
                
        self.perception = perception

        # Status Attributes
        self.happiness = 1.0
        self.fatigue = 0.0

        self.hunger_max = base_hunger
        self.hunger = self.hunger_max
        self.hunger_loss = (floor(self.ATK/10) + floor(self.DEF/10) + floor(self.MHP/5)) / 3
        self.thirst_max = base_thirst
        self.thirst = self.thirst_max
        self.thirst_loss = (floor(self.MAG/10) + floor(self.RES/10) + floor(self.MSP/5) + self.SPD) / 5
                
        # Flavor and Preferences
        self.flavor = flavor
        self.fPrefs = {
            f: random.uniform(0.0, 1.0) * compatibility(self.flavor, f)
                for f in FLAVOR
        }

        # Rare Allergy
        if (random.randint(1, 1000) == 1000):
            allergic = random.choice(FLAVOR[:-1])
            self.fPrefs[allergic] *= -1
        
        # Stats
        self.ATK = 10
        self.DEF = 10
        self.MHP = 50
        self.cHP = self.MHP
        self.MAG = 10
        self.RES = 10
        self.MSP = 50
        self.cSP = self.MSP
        self.SPD = 2

        # Age
        self.age = 0
        self.level = 1
        
        # Achievement Attributes
        self.times_eaten = 0
        self.times_drank = 0
        self.steps_in_pain = 0
        self.was_in_pain = False
        self.happiness_max = 1.0        # Maximum Happiness ever recorded
        self.happiness_min = 1.0        # Minimum Happiness ever recorded
        self.fatigue_max = 0.0          # Highest Fatigue ever recorded
        self.happiness_total = 0.0      # Total Lifetime Happiness
        self.fatigue_total = 0.0        # Total Lifetime Fatigue
        self.average_fatigue = 0.0      # Average Fatigue

        self.death_step = None          # Deathdate
        
        # Movement Attributes
        self.visited = set()
        self.last_action = None
        self.last_failed = False
        self.last_event = None
        self.repeat_event = 0
        self.steps = 0
        self.tile_under = EMPTY
        self.times_bored = 0

        # input_size now guaranteed to match perceive() output
        input_size = self.sight_range * self.cone_width
        hidden_size = 32
        action_size = 9  # 8 directions + sit
        
        # Brain Stuff
        self.local_memory = []

        # Log Birth Message
        print(f"🧠 '{self.name}' was born at ({self.x}, {self.y}) Flavor: {self.flavor}")
 
        
    # --- Perception ---
    def perceive(self):
        """
        Raymarch outward from the agent within the cone of vision.
        Returns a 1D array of numeric perception values.
        """
        pDirections = [(0, -1),   # N
            (1, -1),   # NE
            (1, 0),    # E
            (1, 1),    # SE
            (0, 1),    # S
            (-1, 1),   # SW
            (-1, 0),   # W
            (-1, -1),  # NW
            ]
        pCount = len(pDirections)

        pDepth = 11   # Type + Quality + Flavor Count (including Bland)

        external = np.zeros((pCount, pDepth), dtype=np.float32)

        for d in range(pCount):
            external[d] = np.zeros(pDepth)

        internal = np.array([self.cHP / self.MHP, 
                             self.cSP / self.MSP,
                             self.thirst / self.thirst_max,
                             self.hunger / self.hunger_max,
                             self.happiness / self.happiness_max,
                             -1.0 if self.cHP <= self.MHP * self.pain_threshold else 0.0,
                             self.fatigue / self.average_fatigue,
                             self.steps_in_pain / -500,
                             self.steps / 1000,
                             self.happiness_total / 500,
                             self.times_eaten / 500,
                             self.times_drank / 500,
                             -1.0 if self.last_failed else 0.0], dtype=np.float32)


        # Combine, Normalize, and Return Observation
        obs = np.concatenate([external.flatten(), internal])
        return obs
    
    # --- Decision Making ---
    def decide(self, obs, debug=False):
        """
        Decide on an action based on the observation vector.
        Produces a policy distribution and samples one action.
        Optionally prints a debug summary of perception -> action mapping.
        """

        # --- Network inference ---
        policy, value = self.trainer.network(obs, training=False)

        policy = tf.nn.softmax(policy[0]).numpy()  # Proper softmax normalization
        value = float(value[0].numpy())            # Scalar value

        # --- Normalize probabilities ---
        total = np.sum(policy)
        if total == 0 or np.isnan(total):
            policy = np.ones_like(policy) / len(policy)
        else:
            policy /= total

        # --- Penalize undesirable repeats ---
        if self.last_failed and self.last_action is not None:
            policy[self.last_action] *= 0.25
            policy = policy / np.sum(policy)
        elif self.last_action == len(self.DIRECTIONS):  # action 8 = sit
            policy[self.last_action] *= 0.5
            policy = policy / np.sum(policy)

        # --- Sample action ---
        action = np.random.choice(len(policy), p=policy)

        # --- Optional Debugging ---
        if debug:
            action_names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "SIT"]
            chosen = action_names[action]

            # interpret obs roughly if you know what obs encodes
            obs_str = np.array2string(obs.numpy(), precision=2, separator=", ")

            print(f"\n🤖 Agent {self.name} Decision Step")
            print(f"Observation: {obs_str}")
            print(f"Policy: {[round(p, 3) for p in policy]}")
            print(f"→ Chose action: {chosen} (index {action}) | Value Estimate: {value:.3f}")

            if self.last_failed:
                print(f"⚠️ Last action {action_names[self.last_action]} failed — penalized.")

        # --- Store state ---
        self.last_action = action
        self.last_failed = False  # will be set true if action fails elsewhere

        return action, value

    # --- Movement ---
    def move(self, action):
        event = None

        # Whole move phase is within a Try-Finally block so that Biostasis always runs and always runs once.
        try:
            # No movement = idle
            if action >= len(self.DIRECTIONS):  # Sit still
                self.sit_counter = getattr(self, "sit_counter", 0) + 1
                if self.sit_counter > 3:
                    self.boredom = 0.15 * self.sit_counter
                self.happiness *= 1.0 - getattr(self, "bordom", 0)
                event = "idle"
                return self.compute_reward(event)

            # If we're not idle
            else:
                self.dir = (self.dir + action) % len(self.DIRECTIONS)
                dx, dy = self.DIRECTIONS[action]        
                nx, ny = self.x + dx, self.y + dy
                    

                # Bounds Checking
                # Bump if Target OOB
                if not (0 <= nx < self.grid.width and 0 <= ny < self.grid.height):
                    self.sit_counter = getattr(self, "sit_counter", 0) + 1
                    self.happiness *= 0.95
                    
                    event = "bump"
                    self.last_failed = True
                    return self.compute_reward(event)

                # We have now confirmed a move will occur
                self.tile_under = self.grid.cells[ny][nx]
                cell = self.grid.cells[ny][nx]

                # Pick an action based on target
                # Eat a Food
                if cell == FOOD:
                    self.hunger = self.hunger_max
                    self.happiness *= 1.1
                    self.sit_counter = 0

                    event = "eat"
                    self.grid.cells[ny][nx] = EMPTY
                    self.times_eaten += 1

                # Eat a Corpse
                elif cell == CORPSE:
                    self.hunger = self.hunger_max * .5
                    self.happiness *= .5
                    self.sit_counter = 0

                    event = "eat"
                    print(f"Agent {self.name} ate a Corpse")
                    self.grid.cells[ny][nx] = GRAVE
                    self.tile_under = GRAVE
                    self.times_eaten += 1

                # Drink Water
                elif cell == WATER:
                    # The Agent enters the water and drinks
                    if self.tile_under != WATER:  # Only trigger when first stepping into water
                        self.thirst = self.thirst_max
                        self.happiness *= 1.05
                        self.sit_counter = 0
                        event = "drink"
                        self.times_drank += 1

                    # The Agent is already in water, stop
                    else:
                        nx, ny = self.x, self.y  # Can't go deeper
                        self.sit_counter = getattr(self, "sit_counter", 0) + 1
                        event = "idle"
                        self.happiness *= 0.95  # Slight boredom penalty
                        return self.compute_reward(event)

                # Walk into Danger
                elif cell == DANGER:
                    self.hurt(3)
                    self. sit_counter = 0
                    event = "danger"

                # Agent Collision
                elif cell == AGENT:
                    # Engage in combat
                    target = self.grid.get_agent_at(nx, ny)  # You'll need to add this helper in Grid (shown below)

                    # If the target is valid
                    if target and target.alive:
                        # Reproduction Logic is going in here shortly
                        event = "fight"
                        self.fight(target)
                    else:
                        # If no valid target found (shouldn’t happen)
                        event = "bump"
                    nx, ny = self.x, self.y  # Stay in place after fighting
                    self.sit_counter = getattr(self, "sit_counter", 0) + 1

                # Any other object is impassible and uninteractible
                elif cell != EMPTY:
                    nx, ny = self.x, self.y
                    self.sit_counter = getattr(self, "sit_counter", 0) + 1
                    self.happiness *= 0.9  # Slight decrease for failed move

                    event = "bump"      
        
                # If you made it through all that, it's an empty square        
                else:
                    self.happiness *= 1.1  # Slight increase for successful move
                    event = "move"

                    if (nx, ny) not in self.visited:
                        self.visited.add((nx, ny))
                        self.happiness *= 1.5  # Slight increase for exploring

                    # Move agent
                    self.grid.cells[self.y][self.x] = self.tile_under
                    self.x, self.y = nx, ny
                    self.grid.cells[self.y][self.x] = AGENT
            
                self.last_failed = (event == "bump")
                self.last_action = action
                
                #if event in ["eat"]:
                #    print(f"Agent {self.name} ate(x={self.x}, y={self.y}), Hunger: {self.hunger}, Happiness: {self.happiness:.2f}")
                #elif event in ["drink"]:
                #    print(f"Agent {self.name} drank(x={self.x}, y={self.y}), Thirst: {self.thirst}, Happiness: {self.happiness:.2f}")
            

                if self.is_dead():
                    self.grid.mark_corpse(self.x, self.y)
                    event = "death"
                    print(f"💀 Agent {self.name} died at age {self.age} position({self.x}, {self.y})")

                return self.compute_reward(event)
        finally:
            self.biostasis()
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

        if self.fatigue >= 0.8:
            self.hurt(5)
        elif self.fatigue >= 0.5:
            self.hurt(3)
            
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
        reward = 0.1

        if event == self.last_event:
            self.repeat_events += 1
            reward -= 0.01 * self.repeat_events  # Small penalty for repeating same event
        else:
            self.repeat_events = 0

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
                print(f"💀 Agent {self.name} died in battle at age {self.age} position({self.x}, {self.y})")
            else:
            # Encourage victorious combat, discourage wasteful fights
                reward += 0.5 if self.happiness > 1.0 else -0.2

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
        
        if reward > 0:
            reward *= self.happiness
        else:
            reward /= self.happiness

        # Keep reward within sane range
        reward = np.clip(reward, -1.0, 1.0)
        
        # Set Last Event
        self.last_event = event
        # print (f"Agent {self.name}, Action: {event}, Reward: {reward:.1f}, Happiness: {self.happiness:.3f}")
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
        # Check if damage is above damage threshold (ignore DT for healing, i.e. -damage)
        if (damage > 0 and damage > self.damage_threshold) or (damage < 0):
            self.health -= damage

        # Bounds Checking
        if self.health < 0:
            self.health = 0
            self.is_dead(force=True)
        elif self.health > self.health_max:
            self.health_max += 1
            self.health = self.health_max
        return self.health

    # --- Eat Function ---

    # --- Combat Function ---
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
    
    
    # --- Reset Agent ---
    def reset(self):
        # Remove agent’s old position if needed
        if self.grid.cells[self.y][self.x] == AGENT:
            self.grid.cells[self.y][self.x] = EMPTY

        # Restore stats
        self.hunger = self.hunger_max
        self.thirst = self.thirst_max
        self.health = self.health_max
        self.fatigue = 0.0
        self.happiness = 1.0
        self.sit_counter = 0
        self.alive = True
        self.times_eaten = 0
        self.times_drank = 0
        self.steps_in_pain = 0
        self.was_in_pain = False
        self.death_step = None
        self.age = 0


        # Reset memory and temporary learning buffers
        self.local_memory.clear()
        self.visited.clear()

        # Place back on the grid
        self.grid.cells[self.y][self.x] = AGENT

    # --- Misc Setters and Getters ---    
    # --- Set Trainer ---
    def set_trainer(self, trainer):
        """
        Assign a centralized trainer to the agent.
        """
        self.trainer = trainer

    # --- Set Name ---
    def set_name(self, name):
        """
        Assign a string as Name
        """
        self.name = name

    # --- Get Stats ---
    def get_Stats(self):
        return [self.MHP, self.ATK, self.DEF, self.MSP, self.MAG, self.RES, self.SPD]

    # --- Locally Normalized ---
    def get_lStats(self):
        return [self.MHP/5, self.ATK, self.DEF, self.MSP/3, self.MAG, self.RES, self.SPD*5]

    # --- Combat Normalized ---
    def get_cStats(self):
        cStats = [self.MHP/500.00, self.ATK/99.00, self.DEF/99.00, self.MSP/250.00, self.MAG/99.00, self.RES/99.00, self.SPD / 15]
        return np.normalize(cStats)

    # -- Get Physical Stats ---
    # return MHP, ATK, and DEF values
    def get_Phys(stats):
        return stats[0, 1, 2]

    # --- Get Mental Stats ---
    # return MSP, MAG, and RES values
    def get_Ment(stats):
        return stats[3, 4, 5]

    # --- Get Stamina Stats ---    
    # return MHP, MSP, and SPD values
    def get_Stmn(stats):
        return stats[0, 3, 6]

    # --- Get Offensive Stats ---
    # return ATK, MAG, SPD
    def get_Offn(stats):
        return stats[1, 4, 6]

    # --- Get Defensive Stats ---
    # return MHP, DEF, MSP, RES
    def get_Defn(stats):
        return stats[0, 2, 3, 5]

    # --- Get Max Stat ---
    def get_xStat(self):
        return max(self.get_stats())

    # --- Get Max Offensive ---
    # *Combat Normalized for convenience
    def get_xOffn(self):
        offn = Agent.get_Offn(self.get_cStats())
        return max(offn)
