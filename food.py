# Food Class
# Author:   K. E. Brown
# First:    2025-10-26
# Updated:  2025-10-26

# Imports
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt

# Import shared symbols and grid reference
class Food:
    # Flavor Subsection
    FLAVOR = [ 
        'Neutral',
        'Crunchy',
        'Fatty',
        'Sweet',
        'Sour',
        'Bitter',
        'Spicy',
        'Savory',
        ]
    
    # Define Flavor Compatibility for Linear Interpolation
    def compatibility(flavor1, flavor2):
        # Neutral is mid
        if (flavor1 == 'Neutral' or flavor2 == 'Neutral'):
            return 0.75
        elif (flavor1 == flavor2):
            return 1.0

        # Otherwise, there is an interaction between the two flavors based on distance
        else:
            # Start by defining two distance vector
            distance = abs(Food.FLAVOR.index[flavor2] - Food.FLAVOR.index[flavor1])

            if (distance > 3):
                distance = 7 - distance

            return {1: 0.9, 2: 0.5, 3: 0.1}.get(distance, -1.0)

    def __init__(self, name="Tofu", icon='e', nValue=5, flavor='Neutral', isSafe=True):
        self.name = name
        self.icon = icon
        self.nValue = nValue
        self.flavor = flavor
        self.isSafe = isSafe