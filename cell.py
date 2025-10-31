# Cell Script
# Author:   K. E. Brown, Chad GPT.
# First:    2025-10-27
# Updated:  2025-10-27

# Congrations, Individual Cells are now complex enough that they warrant an entire file

# Let's design a new cell structure that will be beneficial later on.
# Cell:

#	type: land / water
#   contents:   Agent, Food, Water, etc.
#	name
#   clarity
#   walkable, swimable


class Cell:
    CELL_TYPE = [
        'LAND',
        'WATER']
        
    CONT_TYPE = [
        'EMPTY',
        'FOOD',
        'AGENT',
        'TERRAIN']
        
    def __init__(self, name="Cell", cType=CELL_TYPE[0], icon=' ', contents=None, clarity = 5, elevation=1, passable=True):
        self.name = name
        self.cType = cType
        self.icon = icon
        self.contents = contents        
        self.clarity = clarity
        self.elevation = elevation
        self.passable = passable

        # If the cell is not empty, do anything else that needs done
        if self.contents != None:
            return -1
        
        
    def set_conts(self, contents, override=False):
        # Can't Place on Occupied Cell without override
        if self.contents != None and not override:
            return False

        # If the Cell is Impassible, it can't have an object
        elif self.passable == False:
            return False

        # Actually set the contents of an empty Cell
        elif self.contents == None or override:
            self.contents = contents
            return True
        