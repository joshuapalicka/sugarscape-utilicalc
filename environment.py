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

    def __init__(self, size):
        '''
        Constructor
        '''
        (width, height) = size
        self.gridWidth = width
        self.gridHeight = height
        """
        Grid is indexed by: [i][j][0] = sugar capacity (amt currently stored), [i][j][1] = spice capacity (amt currently stored), 
        [i][j][2] = maxSugarCapacity, [i][j][3] = maxSpiceCapacity, [i][j][4] = agent, [i][j][5] = amount of pollution 
        """
        self.grid = [[[0, 0, 0, 0, None, 0.0] for _ in range(width)] for _ in range(height)]
        self.hasSpice = False
        self.hasTags = False
        self.diseases = []
        self.time = 0
        self.idIncrement = 0
        self.loanDuration = 0
        self.loanRate = 0
        self.hasDisease = False
        self.foresight = False
        self.foresightRange = 0, 0
        self.limitedLifespan = True
        self.hasPollution = False
        self.pA = 0
        self.pB = 0
        self.pDiffusionRate = 0
        self.pollutionStartTime = 0
        self.diffusionStartTime = 0

    def getHasSpice(self):
        return self.hasSpice

    def getHasTags(self):
        return self.hasTags

    def setSugarCapacity(self, location, value):
        (i, j) = location
        self.grid[i][j][0] = value

    def setSpiceCapacity(self, location, value):
        (i, j) = location
        self.grid[i][j][1] = value

    def getSugarCapacity(self, location):
        (i, j) = location
        return int(self.grid[i][j][0])

    def getSpiceCapacity(self, location):
        (i, j) = location
        return int(self.grid[i][j][1])

    def decCapacity(self, location, value):
        (i, j) = location
        self.grid[i][j][0] = max(0, self.grid[i][j][2] - value)

    def decSpiceCapacity(self, location, value):
        (i, j) = location
        self.grid[i][j][1] = max(0, self.grid[i][j][3] - value)

    def addSiteHelper(self, location, maxCapacity, sugar):
        # calculate radial dispersion of capacity from maxCapacity to 0
        (si, sj, r) = location
        distance = lambda di, dj: sqrt(di * di + dj * dj)
        D = distance(max(si, self.gridWidth - si), max(sj, self.gridHeight - sj)) * (r / float(self.gridWidth))
        for i, j in product(range(self.gridWidth), range(self.gridHeight)):
            c = min(1 + maxCapacity * (1 - distance(si - i, sj - j) / D), maxCapacity)
            if c > self.grid[i][j][2 if sugar else 3]:
                self.grid[i][j][2 if sugar else 3] = c

    def addSugarSite(self, location, maxCapacity):
        # calculate radial dispersion of capacity from maxCapacity to 0
        self.addSiteHelper(location, maxCapacity, True)

    def addSpiceSite(self, location, maxCapacity):
        self.hasSpice = True
        # calculate radial dispersion of capacity from maxCapacity to 0
        self.addSiteHelper(location, maxCapacity, False)

    def growHelper(self, location, alpha):
        hasSpice = self.getHasSpice()
        (i, j) = location
        self.grid[i][j][0] = min(self.grid[i][j][0] + alpha, self.grid[i][j][2])

        if hasSpice:
            self.grid[i][j][1] = min(self.grid[i][j][1] + alpha, self.grid[i][j][3])

    def grow(self, alpha):
        # grow to maxCapacity with alpha 
        for i, j in product(range(self.gridWidth), range(self.gridHeight)):
            self.growHelper((i, j), alpha)

    def growRegion(self, region, alpha):
        # grow  region to maxCapacity with alpha
        (imin, jmin, imax, jmax) = region
        imin = max(imin, 0)
        jmin = max(jmin, 0)
        imax = min(imax + 1, self.gridWidth)
        jmax = min(jmax + 1, self.gridHeight)
        for j in range(jmin, jmax):
            for i in range(imin, imax):
                self.growHelper((i, j), alpha)

    def setAgent(self, location, agent):
        (i, j) = location
        self.grid[i][j][4] = agent

    def getAgent(self, location):
        (i, j) = location
        return self.grid[i][j][4]

    def isLocationValid(self, location):
        (i, j) = location
        return 0 <= i < self.gridWidth and 0 <= j < self.gridHeight

    def isLocationFree(self, location):
        (i, j) = location
        return self.grid[i][j][4] is None

    def getRandomFreeLocation(self, location):
        # build a list of free locations i.e. where env.getAgent(x,y) == None
        # we don't use a global list and we re-build the list each time 
        # because init a new agent is much less frequent than updating agent's position (that would require append / remove to the global list)
        (xmin, xmax, ymin, ymax) = location
        freeLocations = [(i, j) for i, j in product(range(xmin, xmax), range(ymin, ymax)) if not self.grid[i][j][2]]
        # return random free location if exist
        if len(freeLocations) > 0:
            return freeLocations[random.randint(0, len(freeLocations) - 1)]
        return None

    def incrementTime(self):
        self.time += 1

    def getTime(self):
        return self.time

    def getLoanDuration(self):
        return self.loanDuration

    def getLoanRate(self):
        return self.loanRate

    def setLoanDuration(self, duration):
        self.loanDuration = duration

    def setLoanRate(self, rate):
        self.loanRate = rate

    def getNewId(self):
        self.idIncrement += 1
        return self.idIncrement

    def findAgentById(self, id):
        for i, j in product(range(self.gridWidth), range(self.gridHeight)):
            agent = self.grid[i][j][4]
            if agent is not None and agent.getId() == id:
                return agent
        return None

    def generateDisease(self):
        disease = ""
        for i in range(self.diseaseLength):
            disease += str(random.randint(0, 1))
        self.diseases.append(disease)

    def setDiseaseLength(self, diseaseLength):
        self.diseaseLength = diseaseLength

    def getDiseaseLength(self):
        return self.diseaseLength

    def setImmuneSystemSize(self, size):
        self.immuneSystemSize = size

    def getImmuneSystemSize(self):
        return self.immuneSystemSize

    def setHasDisease(self, hasDisease):
        self.hasDisease = hasDisease

    def getDiseases(self):
        return self.diseases

    def getHasDisease(self):
        return self.hasDisease

    def setHasForesight(self, foresight):
        self.foresight = foresight

    def getHasForesight(self):
        return self.foresight

    def setForesightRange(self, foresightRange):
        self.foresightRange = foresightRange

    def getForesightRange(self):
        return self.foresightRange

    def setHasLimitedLifespan(self, limitedLifespan):
        self.limitedLifespan = limitedLifespan

    def hasLimitedLifespan(self):
        return self.limitedLifespan

    """
        Concept: Pollution

        Follows Agent pollution formation rule P_ab, agent movement rule M, modified for pollution, and pollution diffusion rule D_a 
        from Growing Artificial Societies by Epstein and Axtell, Pgs 47-48

        P_ab: "When sugar quantity s is gathered from the sugarscape, an amount of production pollution is generated in quantity a_s. 
        When sugar amount m is consumed, (metabolized), consumption pollution is generated according to B_m. 
        The total pollution on a site at time t, p&t, is the sum of the pollution present at the previous time, plus the 
        pollution resulting from production and consumption activities, that is, p^t = p^t-1 + a_s + b_m"
        
        M modified for pollution: "Look out as far as vision permits in the four principal lattice directions and identify 
        the unoccupied site(s) having the maximum sugar to pollution ratio; If the maximum sugar to pollution ratio appears 
        on multiple sites, then select the nearest one; Move to this site; Collect all the sugar at this new position"
        
        D_a: "Each a time periods and at each site, compute the pollution flux -- the average pollution level over all 
        von Neumann neighboring sites; Each site's flux becomes its new pollution level"
        
        """

    def setPollutionRules(self, pA, pB, pDiffusionRate, pollutionStartTime, diffusionStartTime):
        self.hasPollution = True
        self.pA = pA
        self.pB = pB
        self.pDiffusionRate = pDiffusionRate
        self.pollutionStartTime = pollutionStartTime
        self.diffusionStartTime = diffusionStartTime

    def polluteSite(self, location, productionPollution, consumptionPollution):
        if self.pollutionStartTime <= self.time:
            (i, j) = location
            self.grid[i][j][5] += (productionPollution * self.pA) + (consumptionPollution * self.pB)

    def diffusePollutionAtLocation(self, x, y):
        neighbourhood = []

        n = x, y + 1
        if self.isLocationValid(n):
            neighbourhood.append(n)

        s = x, y - 1
        if self.isLocationValid(s):
            neighbourhood.append(s)

        e = x + 1, y
        if self.isLocationValid(e):
            neighbourhood.append(e)

        w = x - 1, y
        if self.isLocationValid(w):
            neighbourhood.append(w)

        pollutionAmt = 0
        for location in neighbourhood:
            pollutionAmt += self.grid[location[0]][location[1]][5]

        self.grid[x][y][5] = pollutionAmt / len(neighbourhood)

    def getHasPollution(self):
        return self.hasPollution

    def getPollutionRules(self):
        return self.pA, self.pB, self.pDiffusionRate, self.pollutionStartTime, self.diffusionStartTime

    def spreadPollution(self):
        for i, j in product(range(self.gridWidth), range(self.gridHeight)):
            self.diffusePollutionAtLocation(i, j)

    def getPollutionAtLocation(self, location):
        (i, j) = location
        return self.grid[i][j][5]
