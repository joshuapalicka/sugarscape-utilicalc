'''
Created on 2010-04-18

@author: rv
'''
from math import sqrt
from itertools import product
import random


# helper function to calculate distance for the radial dispersion of sugar/spice
def distance(di, dj):
    return sqrt(di * di + dj * dj)


class Environment:
    def __init__(self, size):

        (width, height) = size
        self.gridWidth = width
        self.gridHeight = height
        self.time = 0

        """
        Grid is indexed by: 
        [i][j][0] = sugar amount (amt currently at site), 
        [i][j][1] = spice amount (amt currently at site) (if spice is not enabled, this is 0 everywhere)
        [i][j][2] = max sugar capacity, 
        [i][j][3] = max spice capacity (if spice is not enabled, this is 0 everywhere)
        [i][j][4] = agent, 
        [i][j][5] = amount of pollution (if pollution is not enabled, this is 0 everywhere)
        """

        self.grid = [[[0, 0, 0, 0, None, 0.0] for _ in range(width)] for _ in range(height)]

        # these are set to True if the rule is enabled. They exist for easy reference by the agents
        self.hasSpice = False
        self.hasTags = False
        self.hasDisease = False
        self.hasPollution = False
        self.hasForesight = False
        self.hasLimitedLifespan = False
        self.hasCombat = False
        self.hasCredit = False

        self.combatAlpha = None

        self.diseases = []
        self.immuneSystemSize = None
        self.diseaseLength = None

        self.idIncrement = 0
        self.loanDuration = 0
        self.loanRate = 0
        self.foresightRange = 0, 0
        self.pA = 0
        self.pB = 0
        self.pDiffusionRate = 0
        self.pollutionStartTime = 0
        self.diffusionStartTime = 0

        self.selfInterestScale = None

        self.maxCapacity = 0

    def getHasSpice(self):
        return self.hasSpice

    def getHasTags(self):
        return self.hasTags

    def getHasCombat(self):
        return self.hasCombat

    def setHasCombat(self, hasCombat):
        self.hasCombat = hasCombat

    def getHasCredit(self):
        return self.hasCredit

    def setHasCredit(self, hasCredit):
        self.hasCredit = hasCredit

    def setMaxCapacity(self, maxCapacity):
        self.maxCapacity = maxCapacity

    def getMaxCapacity(self):
        return self.maxCapacity

    def setSugarAmt(self, location, value):
        (i, j) = location
        self.grid[i][j][0] = value

    def setSpiceAmt(self, location, value):
        (i, j) = location
        self.grid[i][j][1] = value

    def getSugarAmt(self, location):
        (i, j) = location
        return int(self.grid[i][j][0])

    def getSpiceAmt(self, location):
        (i, j) = location
        return int(self.grid[i][j][1])

    def decSugarAmt(self, location, value):
        (i, j) = location
        self.grid[i][j][0] = max(0, self.grid[i][j][2] - value)

    def decSpiceAmt(self, location, value):
        (i, j) = location
        self.grid[i][j][1] = max(0, self.grid[i][j][3] - value)

    def addSiteHelper(self, location, maxCapacity, sugar):
        # calculate radial dispersion of capacity from maxCapacity to 0
        (si, sj, r) = location

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
        (xMin, xMax, yMin, yMax) = location
        freeLocations = [(i, j) for i, j in product(range(xMin, xMax), range(yMin, yMax)) if not self.grid[i][j][4]]
        # return random free location if it exists
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

    # used when getting children of agent, currently this only happens for inheritance
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
        self.hasForesight = foresight

    def getHasForesight(self):
        return self.hasForesight

    def setForesightRange(self, foresightRange):
        self.foresightRange = foresightRange

    def getForesightRange(self):
        return self.foresightRange

    def setHasLimitedLifespan(self, limitedLifespan):
        self.hasLimitedLifespan = limitedLifespan

    def getHasLimitedLifespan(self):
        return self.hasLimitedLifespan

    def setCombatAlpha(self, alpha):
        self.combatAlpha = alpha

    def getCombatAlpha(self):
        return self.combatAlpha

    def setSelfInterestScale(self, selfInterestScale):
        self.selfInterestScale = selfInterestScale

    def getSelfInterestScale(self):
        return self.selfInterestScale

    def getNeighborhood(self, x, y):
        neighbourhood = [self.getAgent((newX, y)) for newX in range(x - 1, x + 2)
                         if self.isLocationValid((newX, y))
                         and not self.isLocationFree((newX, y))]

        neighbourhood.extend([self.getAgent((x, newY)) for newY in range(y - 1, y + 2)
                              if self.isLocationValid((x, newY))
                              and not self.isLocationFree((x, newY))])

        random.shuffle(neighbourhood)

        return neighbourhood

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
