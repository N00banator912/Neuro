# Grid Script
# Author:   K. E. Brown, Chad GPT.
# First:    2025-10-03
# Updated:  2025-10-11

# Imports
from turtle import position
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
TREE    = "T"
MOUNTAIN = "^"
GRAVE = "t"
HOUSE = "h"
LUMBER = "="

# Parameters
DangerCount = 3
init_food_density = 0.7  # Percentage of grid cells to fill with food at init

class Grid:
    def __init__(self, width=25, height=25, seed=69420):
        self.width = width
        self.height = height
        self.cells = [[EMPTY for _ in range(width)] for _ in range(height)]
        self.food_pos = []
        self.danger_pos = []
        self.agents = []
        random.seed(seed)

    def init(self):
        """Generate Perlin noise terrain with natural-looking water, trees, and mountains."""
        scale = 8.0          # higher scale = smaller features
        octaves = 8
        persistence = 0.5
        lacunarity = 2.0

        # --- Generate elevation map ---
        elevation = np.zeros((self.height, self.width))
        for y in range(self.height):
            for x in range(self.width):
                nx = x / self.width - 0.5
                ny = y / self.height - 0.5
                noise_val = pnoise2(nx * scale, ny * scale,
                                    octaves=octaves,
                                    persistence=persistence,
                                    lacunarity=lacunarity,
                                    repeatx=1024, repeaty=1024, base=random.randint(0, 9999))
                elevation[y][x] = (noise_val + 1) / 2.0  # normalize 0–1

        # Optional post-blur to smooth small patches
        elevation = self._smooth_noise(elevation, passes=1)

        # --- Terrain thresholds ---
        for y in range(self.height):
            for x in range(self.width):
                val = elevation[y][x]
                if val < 0.47:
                    self.cells[y][x] = WATER
                elif val < 0.55:
                    self.cells[y][x] = EMPTY
                elif val < 0.6:
                    self.cells[y][x] = TREE
                else:
                    self.cells[y][x] = MOUNTAIN

        # --- Place food ---
        food_count = int(self.width * self.height * init_food_density)
        # print (f"Placing {food_count} food items.")
        self.food_pos = []
        for _ in range(int(food_count/5)):
            self.spawn_food(position=self._rand_pos())

        # --- Place dangers ---
        self.danger_pos = []
        for _ in range(DangerCount):
            self._place_feature(DANGER)

    # --- Helper functions ---
    def _smooth_noise(self, arr, passes=1):
        """Apply a simple smoothing filter to make lakes less blotchy."""
        for _ in range(passes):
            new = arr.copy()
            for y in range(1, self.height - 1):
                for x in range(1, self.width - 1):
                    new[y][x] = (
                        arr[y][x] +
                        arr[y - 1][x] + arr[y + 1][x] +
                        arr[y][x - 1] + arr[y][x + 1]
                    ) / 5.0
            arr = new
        return arr

    def _rand_pos(self):
        return (random.randint(0, self.width - 1), random.randint(0, self.height - 1))

    def _place_feature(self, symbol):
        """Try to place a special feature (like danger) in an empty spot."""
        for _ in range(100):
            x, y = self._rand_pos()
            if self.cells[y][x] == EMPTY:
                self.cells[y][x] = symbol
                return (x, y)
        return None

    def populate(self, agent_list):
        """Place agents randomly in walkable terrain."""
        for agent in agent_list:
            while True:
                x, y = self._rand_pos()
                if self.cells[y][x] == EMPTY:
                    agent.x, agent.y = x, y
                    self.cells[y][x] = AGENT
                    agent.grid = self
                    break

    def get_agent_at(self, x, y):
        """
        Returns the agent object at the given coordinates, if any.
        """
        for agent in self.agents:
            if agent.x == x and agent.y == y and agent.alive:
                return agent
        return None


    def spawn_food(self, count=5, position=[0, 0], radius=2):
        ax, ay = position
        for _ in range(count):
            fx = random.randint(max(0, ax - radius), min(self.width - 1, ax + radius))
            fy = random.randint(max(0, ay - radius), min(self.height - 1, ay + radius))
            if self.cells[fy][fx] == EMPTY:
                self.cells[fy][fx] = FOOD

    def mark_corpse(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.cells[y][x] = CORPSE

    def render(self, hard_print=False):
        os.system('cls' if os.name == 'nt' else 'clear')
        for row in self.cells:
            print("".join(row))
            
    def reseed(self, seed):
        random.seed(seed)