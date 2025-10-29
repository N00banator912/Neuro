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

# Local Imports
from cell import Cell
from food import Food

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
        self.cells = [[Cell() for _ in range(width)] for _ in range(height)]
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
                self.cells[x][y].elevation = (noise_val + 1) / 2.0  # normalize 0–1

        # Optional post-blur to smooth small patches
        elevation = self._smooth_noise(elevation, passes=1)

        # --- Terrain Thresholds ---
        for y in range(self.height):
            for x in range(self.width):
                local_elevation = self.cells[x][y].elevation
                # Oceans and Land
                if local_elevation < 0.4:
                    self.cells[x][y].type = 'WATER'
                    self.clarity = 3
                else:
                    self.cells[x][y].type = 'LAND'

                # Terrain Object Placement
                # Mountains
                if local_elevation > .75:
                    self.cells[x][y].cType = 'TERRAIN'  # This should actually be setting it to a Mountain Object
                
                # Forests
                elif 0.65 < local_elevation < .69:
                    self.cells[x][y].cType = 'TERRAIN'




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


    def place_food(self, food=None, position=[0,0], count=5, radius=3):
        """
        Place one or more Food objects near a given position.
        If no Food object is provided, a default is created.
        """
        x, y = position
        # Create a default Food object if none given
        if food is None:
            food = Food()

        # Find nearby empty cells

        empties = self.find_nearby_empty(x, y, radius=radius, count=count)

        # Place food objects
        placed = 0
        for (nx, ny) in empties:
            cell = self.cells[ny][nx]
            # --- If using symbol-based grid ---
            if isinstance(cell, str):
                self.cells[ny][nx] = FOOD
            # --- If using Cell objects ---
            elif hasattr(cell, "set_conts"):
                cell.set_conts(food)
            placed += 1
            if placed >= count:
                break

        return placed

    def mark_corpse(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.cells[y][x] = CORPSE

    def render(self, hard_print=False):
        os.system('cls' if os.name == 'nt' else 'clear')
        for row in self.cells:
            print("".join(row))
            
    def reseed(self, seed):
        random.seed(seed)

    # Helper Functions
    def find_nearby_empty(self, x, y, radius=1, count=0):
        """
        Returns a list of (x, y) coordinates of empty cells within the given radius.
        Includes diagonals, does not include the original (x, y).
        """
        # Create List of Empty Cells
        empties = []
        # Loop Cells in valid area
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                # Skip the Origin
                if (dx == 0 and dy == 0):
                    continue
                                
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    cell = self.cells[ny][nx]
                    if cell.cType == 'EMPTY' and cell.contents == None:
                        empties.append((nx, ny))
                        if count and len(empties) >= count:
                            return empties
        return empties