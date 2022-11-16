'''
Created on 2010-04-18

@author: rv
'''
from math import sqrt
from itertools import product
import random

class Environment:
    '''
    classdocs
    '''
    
    def __init__(self, xxx_todo_changeme):
        '''
        Constructor
        '''
        (width, height) = xxx_todo_changeme
        self.gridWidth = width
        self.gridHeight = height
        self.grid = [[[0, 0, None] for i in range(width)] for j in range(height)]

    def setCapacity(self, xxx_todo_changeme1, value):
        (i, j) = xxx_todo_changeme1
        self.grid[i][j][0] = value
    
    def getCapacity(self, xxx_todo_changeme2):
        (i, j) = xxx_todo_changeme2
        return int(self.grid[i][j][0])

    def decCapacity(self, xxx_todo_changeme3, value):
        (i,j) = xxx_todo_changeme3
        self.grid[i][j][0] = max(0, self.grid[i][j][0] - value)

    def addFoodSite(self, xxx_todo_changeme4, maxCapacity):
        # calculate radial dispersion of capacity from maxCapacity to 0
        (si, sj, r) = xxx_todo_changeme4
        distance = lambda di, dj : sqrt(di*di + dj*dj)
        D = distance(max(si, self.gridWidth - si), max(sj, self.gridHeight - sj)) * (r/float(self.gridWidth))
        for i,j in product(range(self.gridWidth), range(self.gridHeight)):
            c = min(1 + maxCapacity * (1 - distance(si - i, sj - j) / D), maxCapacity)
            if c > self.grid[i][j][1]:
                self.grid[i][j][1] = c

    def grow(self, alpha):
        # grow to maxCapacity with alpha 
        for i,j in product(range(self.gridWidth), range(self.gridHeight)):
            self.grid[i][j][0] = min(self.grid[i][j][0] + alpha, self.grid[i][j][1])

    def growRegion(self, xxx_todo_changeme5, alpha):
        # grow  region to maxCapacity with alpha
        (imin, jmin, imax, jmax) = xxx_todo_changeme5
        imin = max(imin, 0)
        jmin = max(jmin, 0)
        imax = min(imax + 1, self.gridWidth)
        jmax = min(jmax + 1, self.gridHeight)
        for j in range(jmin, jmax):
            for i in range (imin, imax):
                self.grid[i][j][0] = min(self.grid[i][j][0] + alpha, self.grid[i][j][1])

    def setAgent(self, xxx_todo_changeme6, agent):
        (i, j) = xxx_todo_changeme6
        self.grid[i][j][2] = agent

    def getAgent(self, xxx_todo_changeme7):
        (i, j) = xxx_todo_changeme7
        return self.grid[i][j][2]

    def isLocationValid(self, xxx_todo_changeme8):
        (i, j) = xxx_todo_changeme8
        return (i >= 0 and i < self.gridWidth and  j >= 0 and j < self.gridHeight)
    
    def isLocationFree(self, xxx_todo_changeme9):
        (i, j) = xxx_todo_changeme9
        return (self.grid[i][j][2] == None)
        
    def getRandomFreeLocation(self, xxx_todo_changeme10):
        # build a list of free locations i.e. where env.getAgent(x,y) == None
        # we don't use a global list and we re-build the list each time 
        # because init a new agent is much less frequent than updating agent's position (that would require append / remove to the global list)
        (xmin, xmax, ymin, ymax) = xxx_todo_changeme10
        freeLocations = [(i,j) for i,j in product(range(xmin, xmax), range(ymin, ymax)) if not self.grid[i][j][2]]
        # return random free location if exist
        if len(freeLocations) > 0:
            return freeLocations[random.randint(0, len(freeLocations)-1)]
        return None

