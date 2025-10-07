# Grid Script
# Author:   K. E. Brown, Chad GPT.
# First:    2025-10-03
# Updated:  2025-10-06

# Imports
import numpy as np
import random
import os
from noise import pnoise2

# Symbols
EMPTY = " "
AGENT = "O"
CORPSE = "%"
FOOD = "."
WATER = "~"
DANGER = "X"

# Parameters
WaterFactor = 0.45
DangerCount = 3
init_food_density = 0.1
agents = []

class Grid:
    def __init__(self, width=15, height=15):
        self.width = width
        self.height = height
        self.cells = [[EMPTY for _ in range(width)] for _ in range(height)]

        # Object positions
        self.food_pos = None
        self.danger_pos = []

    def init(self):
        """Generate terrain and place agent, food, and dangers."""
        scale = 10.0
        octaves = 4
        persistence = 0.5
        lacunarity = 2.0

        # --- Generate Perlin Noise terrain ---
        for j in range(self.height):
            for i in range(self.width):
                nx = i / self.width
                ny = j / self.height
                noise_val = pnoise2(nx * scale, ny * scale,
                                    octaves=octaves,
                                    persistence=persistence,
                                    lacunarity=lacunarity,
                                    repeatx=1024, repeaty=1024, base=0)
                noise_val = (noise_val + 1) / 2.0  # normalize to [0,1]
                self.cells[j][i] = WATER if noise_val < WaterFactor else EMPTY

        # --- Place Food ---
        food_count = int(self.width * self.height * init_food_density)
        self.food_pos = []
        self.spawn_food(count=food_count, position=[
            random.randrange(int(self.width / 4), int(self.width * 3 / 4)),
            random.randrange(int(self.height / 4), int(self.height * 3 / 4))])

        # --- Place Dangers ---
        for _ in range(DangerCount):
            attempts = 0
            max_attempts = 50
            while attempts < max_attempts:
                dx, dy = self._rand_pos()
                if self.cells[dy][dx] == EMPTY:
                    self.danger_pos.append((dx, dy))
                    self.cells[dy][dx] = DANGER
                    break
                attempts += 1
                

    # --- Place Agent list (passed from Neuro.py) ---
    def populate(self, agent_list):
        """Place agents on the grid."""
        global agents
        agents = agent_list
        for agent in agents:
            ax, ay = self._place_random(EMPTY)
            agent.x, agent.y = ax, ay
            self.cells[ay][ax] = AGENT
            agent.grid = self           

    def _rand_pos(self):
        """Get a random (x, y) coordinate."""
        return (random.randint(0, self.width - 1), random.randint(0, self.height - 1))

    def _place_random(self, cell_type):
        """Place an object in a random cell of the given type."""
        while True:
            pos = self._rand_pos()
            if self.cells[pos[1]][pos[0]] == cell_type:
                return pos
    
    def spawn_food(self, count=5, position=[0,0], radius=2):
        """Spawn food items around a given position within a specified radius."""
        ax, ay = position
        spawned = 0
        attempts = 0
        max_attempts = count * 5
        while spawned < count and attempts < max_attempts:
            fx = random.randint(max(0, ax - radius), min(self.width - 1, ax + radius))
            fy = random.randint(max(0, ay - radius), min(self.height - 1, ay + radius))
            if self.cells[fy][fx] == EMPTY:
                self.cells[fy][fx] = FOOD
                spawned += 1
            attempts += 1   
    
    def mark_corpse(self, x, y):
        """Mark a cell as containing a corpse when an agent dies."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.cells[y][x] = CORPSE
    
    # Rendering Functions   
    def render(self, hard_print = False):
        """Draw the grid in the console."""
        os.system('cls')
        for row in self.cells:
            print("".join(row))
