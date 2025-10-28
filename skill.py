# Skill Script
# Author:   K. E. Brown, Chad GPT.
# First:    2025-10-28
# Updated:  2025-10-28

from food import FLAVOR

class Skill:
    SKILL_TYPE = []
        
        
    def __init__(self, name="Move"):
        self.name = name
        
        
class Damage:
    def __init__(self, flavor='Neutral', magnitude=1):
        self.flavor = flavor
        self.magnitude = magnitude