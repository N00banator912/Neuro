# Skill Script
# Author:   K. E. Brown, Chad GPT.
# First:    2025-10-28
# Updated:  2025-10-28

from food import FLAVOR

class Skill:
    SKILL_TYPE = [
        'Physical',
        'Magical',
        'Support']
        
        
    def __init__(self, name="Move", damage=None, effects=None):
        self.name = name
        self.damage = damage
        self.effects = effects
        
        
class Damage:
    def __init__(self, flavor='Neutral', magnitude=1):
        self.flavor = flavor
        self.magnitude = magnitude