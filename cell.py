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
        
    def __init__(self, name="Cell", cType=CELL_TYPE[0], contents=None, clarity = 5, passable=True):
        self.name = name
        self.cType = cType
        self.contents = contents
        
        # If the cell is not empty, do anything else that needs done
        if self.contents != None:
            return -1
        
        self.clarity = clarity
        self.passable = passable
        
    def set_conts(self, conents, override=False):
        if self.contents != None and not override:
            return False
        elif self.contents == None or override:
            self.contents = contents
            return True