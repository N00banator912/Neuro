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
            distance = abs(flavor2 - flavor1)

            # Edge Cases (Crunchy, Savory)
            if distance > 3:
                distance = 7 - distance

            # Similar Flavors
            if distance == 1:
                return 0.9
            # Nearby Flavors
            elif distance == 2:
                return 0.5
            # Distant Flavors
            elif distance == 3:
                return 0.1
            # Error Checker
            else:
                return -1.0

    def __init__(self, nValue=5, name="Tofu", flavor='Neutral', isSafe=True):
        self.name = name
        self.nValue = nValue
        self.flavor = flavor
        self.isSafe = isSafe