'''
Created on 2010-04-21

@author: rv
'''
import math
import random
import sys
from itertools import product

import felicific_calculus as fc

"""
  Creates a loan from other agent to self
"""


def createNewLoan(lender, borrower, sugar, spice, time=None, transfer=True):
    if time is None:
        time = borrower.env.getLoanDuration()
    hasSpice = borrower.env.getHasSpice()
    lender.lent.append((lender, borrower, sugar, spice, time))

    # transfer means that goods are transferred from lender to borrower. It is False when the loan is created from another loan's default
    if transfer:
        lender.setSugar(lender.sugar - sugar, "loan")
        if hasSpice:
            lender.setSpice(lender.spice - spice, "loan")

    logEntry = "Created loan of " + str(sugar) + " sugar and " + str(spice) + " spice from " + str(
        lender.getId()) + " to " + str(borrower.getId())

    lender.addLogEntry(logEntry)
    rate = lender.env.getLoanRate()
    borrower.borrowed.append((borrower, lender, sugar * rate, spice * rate, time))

    if transfer:
        borrower.setSugar(borrower.sugar + sugar, "loan")
        if hasSpice:
            borrower.setSpice(borrower.spice + spice, "loan")

    borrower.addLogEntry(logEntry)


def tradeHelper(A, B, p, A_welfare, B_welfare):
    # in this function, A is always the higher MRS agent so spice flows from A to B and sugar flows from B to A
    if p == 1:
        return False

    if p > 1:
        B_potential_sugar = B.getSugar() - 1
        A_potential_sugar = A.getSugar() + 1

        A_potential_spice = A.getSpice() - p
        B_potential_spice = B.getSpice() + p

    elif p < 1:
        B_potential_sugar = B.getSugar() - 1 / p
        A_potential_sugar = A.getSugar() + 1 / p

        A_potential_spice = A.getSpice() - 1
        B_potential_spice = B.getSpice() + 1

    new_A_mrs = A.getMRS(A_potential_sugar, A_potential_spice)
    new_B_mrs = B.getMRS(B_potential_sugar, B_potential_spice)

    new_A_welfare = A.getWelfare(A_potential_sugar, A_potential_spice)
    new_B_welfare = B.getWelfare(B_potential_sugar, B_potential_spice)

    if new_A_welfare > A_welfare and new_B_welfare > B_welfare:
        if new_A_mrs > new_B_mrs:
            A_prev_sugar = A.getSugar()
            A_prev_spice = A.getSpice()
            B_prev_sugar = B.getSugar()
            B_prev_spice = B.getSpice()

            A.setSugar(A_potential_sugar, "Trade")
            B.setSugar(B_potential_sugar, "Trade")
            A.setSpice(A_potential_spice, "Trade")
            B.setSpice(B_potential_spice, "Trade")

            A_trade_str = "I traded with Agent " + str(B.getLocation()) + " A started with " + str(
                A_prev_sugar) + " sugar and " + str(A_prev_spice) + " spice. They ended with " + str(
                A.getSugar()) + " sugar and " + str(A.getSpice()) + " spice. Cost was: " + str(p)
            B_trade_str = "I traded with Agent " + str(A.getLocation()) + " B started with " + str(
                B_prev_sugar) + " sugar and " + str(B_prev_spice) + " spice. They ended with " + str(
                B.getSugar()) + " sugar and " + str(B.getSpice()) + " spice. Cost was: " + str(p)

            A.addTrade(A_trade_str)
            B.addTrade(B_trade_str)

            return True
    return False


def updateHelper(disease, immuneSubstr):  # updates immuneSubstr to match disease 1 hamming distance better
    immuneSubstr = list(immuneSubstr)
    disease = list(disease)
    for i in range(len(immuneSubstr)):
        if immuneSubstr[i] != disease[i]:
            if immuneSubstr[i] == '0':
                immuneSubstr[i] = '1'
                return "".join(immuneSubstr)
            else:
                immuneSubstr[i] = '0'
                return "".join(immuneSubstr)


def getDistance(x1, y1, x2, y2):
    return int(math.floor(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)))


class Agent:
    '''
    classdocs
    '''

    tagsProbability = 0.50

    def __init__(self, env, x=0, y=0, sugarMetabolism=4, spiceMetabolism=4, vision=6, sugarEndowment=25,
                 spiceEndowment=25, maxAge=100, sex=0, fertility=(12, 50), tags=(0, 11)):
        '''
        Constructor
        '''

        self.tribe = None
        self.tagsLength = None
        self.tags = None
        self.env = env
        self.id = self.env.getNewId()
        self.x = x
        self.y = y
        self.sugar = 0
        self.spice = 0
        self.sugarMetabolism = sugarMetabolism
        self.spiceMetabolism = spiceMetabolism
        self.vision = vision
        self.maxAge = maxAge
        self.age = 0
        self.sex = sex
        self.fertility = fertility
        self.sugarCostForMatingRange = 0, 0  # 10, 40
        self.spiceCostForMatingRange = 0, 0  # 10, 40
        self.sugarEndowment = None
        self.sugarCostForMating = None
        self.spiceEndowment = None
        self.spiceCostForMating = None
        self.tradeLog = []
        self.sugarLog = []
        self.spiceLog = []
        self.loanLog = []
        self.log = []
        self.lent = []  # list of tuples (lender, borrower, sugar, spice, time)
        self.borrowed = []  # list of tuples (borrower, lender, sugar, spice, time)
        self.setInitialSugarEndowment(sugarEndowment)
        self.setInitialSpiceEndowment(spiceEndowment)
        self.previousWealth = 0
        self.setTags(tags)
        self.immuneSystem = ""
        self.untrainedImmuneSystem = ""
        self.diseases = {}
        self.numAfflictedDiseases = 0
        self.childIds = []
        self.alive = True
        self.foresight = self.generateForesight()

    ''' 
    get / set section 
    '''

    def getEnv(self):
        return self.env

    def setLocation(self, location):
        previousLocation = self.getLocation()
        self.x = location[0]
        self.y = location[1]
        move_str = "Agent put at location: " + str(previousLocation) + " ->" + str(location)
        self.addLogEntry(move_str)

    def getLocation(self):
        return self.x, self.y

    def setSugarMetabolism(self, metabolism):
        self.sugarMetabolism = metabolism

    def setSpiceMetabolism(self, metabolism):
        self.spiceMetabolism = metabolism

    def getSugarMetabolism(self):
        return self.sugarMetabolism

    def getSpiceMetabolism(self):
        return self.spiceMetabolism

    def setVision(self, vision):
        self.vision = vision

    def getVision(self):
        return self.vision

    def setInitialSugarEndowment(self, endowment):
        self.sugarEndowment = endowment
        self.setSugar(endowment, "initial")
        if self.sugarCostForMatingRange == (0, 0):
            self.sugarCostForMating = endowment
        else:
            self.sugarCostForMating = random.randint(self.sugarCostForMatingRange[0], self.sugarCostForMatingRange[1])

    def setInitialSpiceEndowment(self, endowment):
        self.spiceEndowment = endowment
        self.setSpice(endowment, "initial")
        if self.spiceCostForMatingRange == (0, 0):
            self.spiceCostForMating = endowment
        else:
            self.spiceCostForMating = random.randint(self.spiceCostForMatingRange[0], self.spiceCostForMatingRange[1])

    def getSugar(self):
        return self.sugar

    def getSpice(self):
        return self.spice

    def setSugar(self, amount, reason=""):
        previousSugar = self.sugar
        self.sugar = max(amount, 0 if self.env.getHasLimitedLifespan() else .01)
        sugarLog_str = "sugar: " + str(previousSugar) + " -> " + str(self.sugar) + " (" + reason + ")"
        self.sugarLog.append(sugarLog_str)
        self.addLogEntry(sugarLog_str)

    def setSpice(self, amount, reason=""):
        previousSpice = self.spice
        self.spice = max(amount, 0 if self.env.getHasLimitedLifespan() else .01)
        spiceLog_str = "spice: " + str(previousSpice) + " -> " + str(self.spice) + " (" + reason + ")"
        self.spiceLog.append(spiceLog_str)
        self.addLogEntry(spiceLog_str)

    def setAge(self, maxAge):
        self.maxAge = maxAge
        self.age = 0

    def getAge(self):
        return self.age

    def getMaxAge(self):
        return self.maxAge

    def setSex(self, sex):
        self.sex = sex

    def getSex(self):
        return self.sex

    def setFertility(self, fertility):
        self.fertility = fertility

    def setTags(self, tags):
        (tags, tagsLength) = tags
        self.tags = tags
        self.tagsLength = tagsLength
        self.tribe = round(float(bin(tags).count('1')) / float(tagsLength))

    def getTags(self):
        return self.tags

    def getTagsLength(self):
        return self.tagsLength

    def getTribe(self):
        return self.tribe

    def getSugarLog(self):
        return self.sugarLog

    def getSpiceLog(self):
        return self.spiceLog

    def printSugarLog(self):
        for entry in self.sugarLog:
            print(entry)

    def printSpiceLog(self):
        for entry in self.spiceLog:
            print(entry)

    def addTrade(self, trade):
        self.tradeLog.append(trade)
        # self.addLogEntry(trade)  # clutters the log

    def getTrades(self):
        return self.tradeLog

    def printTrades(self):
        for trade in self.tradeLog:
            print(trade)

    def addLogEntry(self, entry):
        time_str = str(self.env.getTime()) + ": "
        self.log.append(time_str + entry)

    def getLog(self):
        return self.log

    def printLog(self):
        for entry in self.log:
            print(entry)

    def getNumAfflictedDiseases(self):
        return len(self.diseases)

    def setUntrainedImmuneSystem(self):
        self.untrainedImmuneSystem = self.immuneSystem

    def getUntrainedImmuneSystem(self):
        return self.untrainedImmuneSystem

    def getWealth(self):
        return self.sugar if not self.env.getHasSpice else self.sugar + self.spice

    def getId(self):
        return self.id

    def isAlive(self):
        return self.alive

    def setAlive(self, alive):
        self.alive = alive

    def generateForesight(self):
        if self.env.getHasForesight:
            foresightRange = self.env.getForesightRange()
            return random.randint(foresightRange[0], foresightRange[1])
        return 0, 0

    def getForesight(self):
        return self.foresight

    def addChildId(self, childId):
        self.childIds.append(childId)

    def getChildren(self):
        children = []
        for childId in self.childIds:
            child = self.env.findAgentById(childId)
            if child is not None:
                children.append(child)
        return children

    def getRandomImmuneSystem(self):
        for i in range(self.env.getImmuneSystemLength()):
            self.immuneSystem += str(random.randint(0, 1))

    def isDiseased(self):
        return len(self.diseases) > 0

    ''' 
    build common lists
    '''

    # build a list of available food locations
    def getFood(self):
        food = [(x % self.env.gridWidth, self.y) for x in range(self.x - self.vision, self.x + self.vision + 1)
                if self.env.isLocationValid((x % self.env.gridWidth, self.y))
                and self.env.isLocationFree((x % self.env.gridWidth, self.y))]

        food.extend([(self.x, y % self.env.gridHeight) for y in range(self.y - self.vision, self.y + self.vision + 1)
                     if self.env.isLocationValid((self.x, y % self.env.gridHeight))
                     and self.env.isLocationFree((self.x, y % self.env.gridHeight))])
        random.shuffle(food)

        return food

    # build a list of possible neighbours for in neighbourhood
    def getNeighbourhood(self):
        neighbourhood = [self.env.getAgent((x % self.env.gridWidth, self.y)) for x in range(self.x - 1, self.x + 2)
                         if self.env.isLocationValid((x % self.env.gridWidth, self.y))
                         and not self.env.isLocationFree((x % self.env.gridWidth, self.y))
                         and x != self.x]

        neighbourhood.extend(
            [self.env.getAgent((self.x, y % self.env.gridHeight)) for y in range(self.y - 1, self.y + 2)
             if self.env.isLocationValid((self.x, y % self.env.gridHeight))
             and not self.env.isLocationFree((self.x, y % self.env.gridHeight))
             and y != self.y])

        random.shuffle(neighbourhood)

        return neighbourhood

    # build a list of neighbours within agent's vision range
    def getVisionNeighbourhood(self):
        neighbourhood = [self.env.getAgent((x % self.env.gridWidth, self.y)) for x in
                         range(self.x - self.vision, self.x + self.vision + 1)
                         if self.env.isLocationValid((x % self.env.gridWidth, self.y))
                         and not self.env.isLocationFree((x % self.env.gridWidth, self.y))
                         and x % self.env.gridWidth != self.x]

        neighbourhood.extend(
            [self.env.getAgent((self.x, y % self.env.gridHeight)) for y in
             range(self.y - self.vision, self.y + self.vision + 1)
             if self.env.isLocationValid((self.x, y % self.env.gridHeight))
             and not self.env.isLocationFree((self.x, y % self.env.gridHeight))
             and y % self.env.gridHeight != self.y])

        random.shuffle(neighbourhood)

        return neighbourhood

    # build a list of possible preys around
    def getPreys(self):
        preys = [self.env.getAgent((x % self.env.gridWidth, self.y)) for x in
                 range(self.x - self.vision, self.x + self.vision + 1)
                 if self.env.isLocationValid((x % self.env.gridWidth, self.y))
                 and not self.env.isLocationFree((x % self.env.gridWidth, self.y))
                 and self.sugar > self.env.getAgent((x % self.env.gridWidth, self.y)).getSugar()
                 and self.env.getAgent((x % self.env.gridWidth, self.y)).getTribe() != self.getTribe()]

        preys.extend([self.env.getAgent((self.x, y % self.env.gridHeight)) for y in
                      range(self.y - self.vision, self.y + self.vision + 1)
                      if self.env.isLocationValid((self.x, y % self.env.gridHeight))
                      and not self.env.isLocationFree((self.x, y % self.env.gridHeight))
                      and self.sugar > self.env.getAgent((self.x, y % self.env.gridHeight)).getSugar()
                      and self.env.getAgent((self.x, y % self.env.gridHeight)).getTribe() != self.getTribe()])

        random.shuffle(preys)

        return preys

    def getFoodWithCombat(self, prey_list):
        preys = prey_list

        food = self.getFood()
        C0 = self.sugar - self.sugarMetabolism
        food.extend([preyA.getLocation() for preyA, preyB in product(preys, preys)
                     if preyA != preyB
                     and preyB.getSugar() < (
                             C0 + self.env.getSugarAmt(preyA.getLocation()) + min(self.env.getCombatAlpha(),
                                                                                  preyA.getSugar()))])

        random.shuffle(food)
        return food

    ''' 
    rules
    '''

    """
    Concept: Credit

    Follows Agent credit rule L_dr from Growing Artificial Societies by Epstein and Axtell, Pgs. 131 & 133

    "An agent is a potential lender if it is too old to have children, in which case the maximum amount it may lend is 
    one-half of its current wealth;

    an agent is a potential lender if it is of childbearing age and has wealth in excess of the amount necessary to have 
    children, in which case the maximum amount it may lend is the excess wealth

    an agent is a potential borrower if it is of childbearing age and has insufficient wealth to have a child and has 
    income (resources gathered, minus metabolism, minus other loan obligations) in the present period making it credit-worthy 
    for a loan written at terms specified by the lender

    If a potential borrower and a potential lender are neighbors then a loan is originated with a duration of d years at 
    the rate of r percent and the face value of the loan amount is transferred from the lender to the borrower

    At the time of the loan due date, if the borrower has sufficient wealth to repay the loan then a transfer from the 
    borrower to the lender is made; else the borrower is required to pay back half of its wealth and a new loan is 
    originated for the remaining sum

    If the borrower on an active loan dies before the due date then the lender simply takes a loss

    if the lender on an active loan dies before the due date then the borrower is not required to pay back the loan, 
    unless inheritance rule I is active, in which case the lender's children now become the borrower's creditors"

    Here, I will only check if each agent can borrow from all others around them, and because everyone gets a turn to 
    move,this will be a fair system.
    """

    def credit(self):
        sugarObligations, spiceObligations = self.checkLoans()
        canBorrow = self.isPotentialBorrower(sugarObligations, spiceObligations)
        if not canBorrow:
            return

        neighbours = self.getNeighbourhood()
        for neighbour in neighbours:
            maxLendSugar, maxLendSpice = neighbour.getPotentialLendAmt()
            sugarNeeded, spiceNeeded = self.amountToReproduce()

            if maxLendSugar >= sugarNeeded and not maxLendSpice >= spiceNeeded:
                createNewLoan(neighbour, self, sugarNeeded, 0)
            elif maxLendSpice >= spiceNeeded and not maxLendSugar >= sugarNeeded:
                createNewLoan(neighbour, self, 0, spiceNeeded)
            elif maxLendSugar >= sugarNeeded and maxLendSpice >= spiceNeeded:
                createNewLoan(neighbour, self, sugarNeeded, spiceNeeded)

    #  get the amount that an agent can lend, according to the book's rules
    def getPotentialLendAmt(self):
        if self.age > self.fertility[1]:
            return .5 * self.sugar, .5 * self.spice

        if self.age > self.fertility[0]:
            excessSugar = max(0, self.sugar - self.sugarCostForMating)
            if not self.env.getHasSpice():
                return excessSugar, 0
            else:
                excessSpice = max(0, self.spice - self.spiceCostForMating)
                return excessSugar, excessSpice
        return 0, 0

    def checkLoans(self):
        for loan in self.lent:
            loan = (loan[0], loan[1], loan[2], loan[3], loan[4] - 1)
            if loan[4] <= 0:
                self.lent.remove(loan)

        sugarObligations, spiceObligations = 0, 0

        for loan in self.borrowed:
            borrower = loan[0]
            lender = loan[1]
            sugarOwed = loan[2]
            spiceOwed = loan[3]
            loan = (loan[0], loan[1], loan[2], loan[3], loan[4] - 1)

            if loan[4] == 0:
                self.borrowed.remove(loan)
                if not lender.isAlive():
                    continue

                #  If their loan duration is up, they must pay back the loan unless they cant afford it
                cantPay = False
                mySugar, mySpice = self.getSugar(), self.getSpice()
                if self.getSugar() - sugarOwed < 0:
                    cantPay = True
                    self.addLogEntry("Agent in debt!")
                    lender.setSugar(lender.getSugar() + mySugar * .5, "repayment, agent in debt")
                    self.setSugar(mySugar * .5, "debt")
                    sugarObligations += mySugar * .5
                    sugarOwed -= mySugar * .5

                if self.env.getHasSpice() and self.getSpice() - spiceOwed < 0:
                    cantPay = True
                    self.addLogEntry("Agent in debt!")
                    lender.setSugar(lender.getSpice() + self.getSpice() * .5, "repayment, agent in debt")
                    self.setSugar(self.getSpice() * .5, "debt")
                    spiceObligations += mySpice * .5
                    spiceOwed -= mySpice * .5

                if cantPay:
                    createNewLoan(lender, self, sugarOwed, spiceOwed, time=self.env.getLoanDuration(), transfer=False)

                else:
                    self.addLogEntry("Agent repaid loan")
                    lender.setSugar(lender.getSugar() + sugarOwed, "repayment")
                    self.setSugar(self.getSugar() - sugarOwed, "repayment")
                    if self.env.getHasSpice():
                        lender.setSpice(lender.getSpice() + spiceOwed, "repayment")
                        self.setSpice(self.getSpice() - spiceOwed, "repayment")
        return sugarObligations, spiceObligations

    # True if agent has income (resources gathered - metabolism - other loans) in current period
    def hasIncome(self, sugarObligations, spiceObligations):
        gathered = self.getWealth() - self.previousWealth
        if self.env.getHasSpice():
            return gathered - self.sugarMetabolism - self.spiceMetabolism - spiceObligations > 0
        else:
            return gathered - self.sugarMetabolism - sugarObligations > 0

    def amountToReproduce(self):
        sugarNeeded = max(0, self.sugarCostForMating - self.sugar)
        if self.env.getHasSpice():
            spiceNeeded = max(0, self.spiceCostForMating - self.spice)
            return sugarNeeded, spiceNeeded
        else:
            return sugarNeeded, 0

    def isPotentialBorrower(self, sugarObligations, spiceObligations):
        if self.fertility[0] <= self.age <= self.fertility[1]:  # fertile
            if self.env.getHasSpice():  # has spice
                if self.sugar < self.sugarCostForMating or self.spice < self.spiceCostForMating:  # not enough resources
                    return self.hasIncome(sugarObligations, spiceObligations)  # has income in current period
            else:  # no spice
                if self.sugar < self.sugarCostForMating:
                    return self.hasIncome(sugarObligations, spiceObligations)  # has income in current period
        return False

    # if inheritance rule I is active, the lender's children now become the borrower's creditors
    def splitLoansAmongChildren(self, children):
        for loan in self.lent:
            borrower = loan[0]
            sugar = loan[2]
            spice = loan[3]
            timeRemaining = loan[4]
            numRecipients = len(children)
            sugar_per_recipient = sugar / numRecipients
            spice_per_recipient = spice / numRecipients
            if borrower in children:
                children.remove(borrower)
            for child in children:
                createNewLoan(child, loan[0], sugar_per_recipient, spice_per_recipient, time=timeRemaining,
                              transfer=False)  # lender, borrower, sugar, spice, time, transfer

    """
    Concept: Cultural Transmission

    Follows Agent cultural transmission (tag-flipping) rule from Growing Artificial Societies by Epstein and Axtell, Pg 73

    "For each neighbor, a tag is randomly selected

    If the neighbor agrees with the agent at that tag position, no change is madel if they disagree, the neighbor's 
    tag is flipped to agree with the agent's tag"
    """

    def transmit(self):
        self.env.hasTags = True
        # build a list of possible neighbours for in neighbourhood
        neighbourhood = self.getNeighbourhood()

        # tag-flipping with neighbours
        for neighbour in neighbourhood:
            mask = 1 << random.randint(0, self.tagsLength - 1)
            neighbourTags = neighbour.getTags()
            if (self.tags & mask) != (neighbourTags & mask):
                # flip neighbour's tag
                neighbourTags ^= mask
                # transmit new tag
                neighbour.setTags((neighbourTags, self.tagsLength))
                self.addLogEntry("transmit: " + str(neighbourTags) + " (" + str(mask) + ")")

    """
    Concept: Aging

    If age > maxAge Then the agent is dead (return False)
    """

    def incAge(self):
        self.age += 1
        self.addLogEntry("age: " + str(self.age))
        return self.maxAge > self.age

    """
    Concept: Sex

    Follows Agent sex rule S from Growing Artificial Societies by Epstein and Axtell, Pg 56

    "Select a neighboring agent at random.

    If the neighbor is fertile and of the opposite sex and at least one of the agent has an empty neighboring site, then a child is born

    Repeat for all neighbors.
    """

    def mate(self):
        # build a list of possible partners in neighbourhood
        neighbourhood = self.getNeighbourhood()

        # mate with (all) possible partners
        for neighbour in neighbourhood:
            # partner selection
            if neighbour.getSex() == self.sex or not neighbour.isFertile():
                continue
            # find a free location around the agent for the baby
            freeLocation = self.findFreeLocationAround(self.x, self.y)
            if not freeLocation:
                # or find around the partner
                freeLocation = self.findFreeLocationAround(neighbour.x, neighbour.y)
            # then, give birth if a location has been found
            if freeLocation:
                if self.isFertile():
                    yield self.createChild(neighbour, freeLocation)

    # Find a free location around x,y (for baby)
    def findFreeLocationAround(self, x, y):
        """locations = [(i, j) for i,j in product(range(x - 1, x + 2), range(y - 1, y + 2))
                     if self.env.isLocationValid((i, j))
                     and self.env.isLocationFree((i, j))]"""
        locations = [(i, y) for i in range(x - 1, x + 2) if
                     self.env.isLocationValid((i, y)) and self.env.isLocationFree((i, y))]
        locations.extend([(x, j) for j in range(y - 1, y + 2) if
                          self.env.isLocationValid((x, j)) and self.env.isLocationFree((x, j))])
        length = len(locations)
        if length:
            return locations[random.randint(0, length - 1)]
        return None

    # Cross-over offspring from two parents: parent1 and parent2
    def createChild(self, parent, childLocation):
        # cross-over parents genetics
        (x, y) = childLocation
        genitors = [self, parent]
        sugarMetabolism = genitors[random.randint(0, 1)].sugarMetabolism
        spiceMetabolism = genitors[random.randint(0, 1)].spiceMetabolism
        vision = genitors[random.randint(0, 1)].vision

        sugarEndowment = 0.5 * (genitors[0].sugarEndowment + genitors[1].sugarEndowment)
        spiceEndowment = 0.5 * (genitors[0].spiceEndowment + genitors[1].spiceEndowment)

        genitors[0].setSugar(max(genitors[0].sugar - 0.5 * genitors[0].sugarEndowment, 0), "Create child")
        genitors[1].setSugar(max(genitors[1].sugar - 0.5 * genitors[1].sugarEndowment, 0), "Create child")

        if self.env.getHasSpice():
            genitors[0].setSpice(max(genitors[0].spice - 0.5 * genitors[0].spiceEndowment, 0), "Create child")
            genitors[1].setSpice(max(genitors[1].spice - 0.5 * genitors[1].spiceEndowment, 0), "Create child")

        if self.env.getHasDisease():
            childImmuneSystem = ""
            genitor0ImmSyst = list(genitors[0].immuneSystem)
            genitor1ImmSyst = list(genitors[1].immuneSystem)
            for i in range(len(genitor0ImmSyst)):
                if genitor0ImmSyst[i] == genitor1ImmSyst[i]:
                    childImmuneSystem += str(genitor0ImmSyst[i])
                else:
                    childImmuneSystem += genitor0ImmSyst[i] if random.randint(0, 1) == 0 else genitor1ImmSyst[i]

        ageMax = genitors[random.randint(0, 1)].maxAge
        sex = random.randint(0, 1)
        fertility = genitors[random.randint(0, 1)].fertility

        # build cultural tags from parents genetics
        mask = 1
        childTags = 0
        for tag in range(self.tagsLength):
            tag1 = self.tags & mask
            tag2 = parent.getTags() & mask
            if tag1 == tag2 or random.random() < self.tagsProbability:
                childTags |= tag1
            else:
                childTags |= tag2
            mask <<= 1

        # create child agent
        self.addLogEntry(str("create child: " + str(childLocation)))

        child = Agent(self.env, x, y, sugarMetabolism, spiceMetabolism, vision, sugarEndowment, spiceEndowment, ageMax,
                      sex, fertility,
                      (childTags, self.tagsLength))

        if self.env.getHasDisease():
            child.immuneSystem = childImmuneSystem
            child.untrainedImmuneSystem = childImmuneSystem

        self.env.setAgent((x, y), child)

        childId = child.getId()
        # create child agent
        self.addLogEntry(str("create child: " + str(childId)))
        parent.addLogEntry(str("create child: " + str(childId)))
        self.addChildId(childId)
        parent.addChildId(childId)
        return parent, child

    """
    Concept: Fertility

    Follows fertility definition from Growing Artificial Societies by Epstein and Axtell, Pg 55

    "First, to have offspring, agents must be of childbearing age.
    Second, children born with literally no initial endowment of sugar would instantly die.
    We therefore require that parents give their children some initial endowment.
    Each newborn's endowment is the sum of the (usually unequal) contributions of mother and father.
    Dad contributes an amount equal to one half of whatever his initial endowment had been, and likewise for mom.
    To be parents, agents must have amassed at least the amount of sugar which they were endowed at birth."
    """

    def isFertile(self):
        if self.fertility[0] <= self.age <= self.fertility[1]:
            if not self.env.getHasSpice():
                if self.sugar >= self.sugarCostForMating:
                    return True
            else:
                if self.sugar >= self.sugarCostForMating and self.spice >= self.spiceCostForMating:
                    return True
        return False

    """
    Concept: Movement

    Follows Agent movement rule M from Growing Artificial Societies by Epstein and Axtell, Pg 25

    "Look out as far as vision permits in the four principal lattice directions and identify the unoccupied site(s) having the most sugar.
    If the greatest sugar value appears on multiple sites then select the nearest one.
    Move to this site.
    Collect all the sugar at this new position.
    Increment the agent's accumulated sugar wealth by the sugar collected and decrement by the agent's metabolic rate."
    
    Also can follow multicommodity agent movement rule M from Growing Artificial Societies by Epstein and Axtell, Pg 98
    
    "Look out as far as vision permits in each of the four lattice directions, north, south, east, and west
    Considering only unoccupied lattice positions, find the nearest position producing maximum welfare
    Move to the new position
    Collect all the resources at that location"
    """

    def move(self):
        # find best food location
        self.previousWealth = self.getWealth()

        move = False
        newx = self.x
        newy = self.y

        # build a list of available food locations
        food = self.getFood()

        locations = []
        food.append((self.x, self.y))

        if not self.env.getHasSpice():
            for (x, y) in food:
                location = (x, y)
                sugarCapacity = self.env.getSugarAmt((x, y))
                distance = self.getManhattanDistance(x, y)  # Manhattan distance enough due to no diagonal
                pollution = self.env.getPollutionAtLocation((x, y))
                locations.append((location, sugarCapacity, distance, pollution))

            locations.sort(key=lambda x: x[2])
            if self.env.getHasPollution():
                best_location = max(locations, key=lambda x: x[1] / (1 + x[3]))  # sugar / 1 + pollution

            else:
                best_location = max(locations, key=lambda x: x[1])

            best_sugar = best_location[1]

            if self.env.getHasPollution():
                self.env.polluteSite(best_location[0], best_sugar, 0)

            if best_location[0] != (self.x, self.y):
                move = True
                newx, newy = best_location[0]

            self.setSugar(max(self.sugar + best_sugar, 0), "Move")
        else:
            for (x, y) in food:
                welfare = self.getWelfare(x=x, y=y)
                location = (x, y)
                sugarCapacity = self.env.getSugarAmt((x, y))
                spiceCapacity = self.env.getSpiceAmt((x, y))
                distance = self.getManhattanDistance(x, y)  # Manhattan distance enough due to no diagonal
                locations.append((welfare, location, sugarCapacity, spiceCapacity, distance))

            locations.sort(key=lambda x: x[4])
            best_location = max(locations, key=lambda x: x[0])  # gets closest location with highest welfare

            if best_location[1] != (self.x, self.y):
                move = True
                newx, newy = best_location[1]

            best_sugar = best_location[2]
            best_spice = best_location[3]

            self.setSpice(max(self.spice + best_spice, 0), "Move")
            self.setSugar(max(self.sugar + best_sugar, 0), "Move")

        # move to new location if any
        if move:
            self.env.setAgent((self.x, self.y), None)
            self.env.setAgent((newx, newy), self)
            previousLocation = (self.x, self.y)
            self.x = newx
            self.y = newy
            self.addLogEntry(str("move: " + str(previousLocation) + " -> " + str((newx, newy))))

        self.env.setSugarAmt((self.x, self.y), 0)
        self.env.setSpiceAmt((self.x, self.y), 0)

    def getManhattanDistance(self, x, y):
        return abs(self.x - x) + abs(self.y - y)

    def getDaysToStarvation(self):
        if not self.env.getHasSpice():
            return self.sugar // self.sugarMetabolism
        else:
            return min(self.sugar // self.sugarMetabolism, self.spice // self.spiceMetabolism)

    def utilicalcPollution(self, agent, pMove):
        circumstances = {}
        pollution = self.env.getPollutionAtLocation(pMove)
        sugarCapacity = self.env.getSugarAmt(pMove)
        circumstances["duration"] -= (sugarCapacity / (1 + pollution)) / agent.sugarMetabolism,  # sugar / 1 + pollution
        return circumstances

    def utilicalcCombat(self, agent,
                        pMove):  # TODO: do something interesting with tags here? Should agent care about agents with other tags?
        circumstances = {}
        agentX, agentY = agent.x, agent.y
        # if we're moving
        if agent == self:
            daysToDeath = agent.getDaysToStarvation()

            agentOnSite = self.env.getAgent((pMove[0], pMove[1]))

            siteWealth = self.env.getSugarAmt((pMove[0], pMove[1]))
            if agentOnSite:
                if agentOnSite.getTags() == agent.getTags():
                    return
                siteWealth += min(agentOnSite.getSugar(), self.env.getCombatAlpha())
            circumstances["intensity"] = 1 / (1 + daysToDeath)
            circumstances["duration"] = siteWealth / agent.sugarMetabolism
            circumstances["certainty"] = 1  # certainty is their distance from the food. 0 if they cannot see the food.

        # if this agent will be killed by a move here
        elif pMove == agent.getLocation():
            daysToDeath = agent.getDaysToStarvation()
            circumstances["intensity"] = 1 - (1 / (1 + daysToDeath))  # agent will be killed
            circumstances["duration"] = 1
            circumstances["certainty"] = 1

        # If they are of the same tribe
        elif agent.getTribe() == self.getTribe():
            daysToDeath = agent.getDaysToStarvation()

            agentOnSite = self.env.getAgent((pMove[0], pMove[1]))
            siteWealth = self.env.getSugarAmt((pMove[0], pMove[1]))

            if agentOnSite and agentOnSite.getTags() != agent.getTags():
                # if it is not a battle the current agent can win, do not consider the move
                if (agent.sugar + self.env.getSugarAmt((pMove[0],
                                                        pMove[1])) - self.sugarMetabolism) <= agentOnSite.sugar:
                    return
                siteWealth += min(agentOnSite.getSugar(), self.env.getCombatAlpha())

            circumstances["intensity"] = 1 / (1 + daysToDeath)
            circumstances["duration"] = siteWealth / agent.sugarMetabolism
            circumstances["certainty"] = 1 if getDistance(agentX, agentY, pMove[0], pMove[
                1]) <= agent.getVision() else 0  # certainty is their distance from the food. 0 if they cannot see the food.

        # If they are of a different tribe
        else:
            agentOnSite = self.env.getAgent(pMove)
            if agentOnSite and agentOnSite.getTags() == agent.getTags():
                daysToDeath = agent.getDaysToStarvation()

                circumstances["intensity"] = daysToDeath
                circumstances["duration"] = .5
                circumstances["certainty"] = 1
        return circumstances

    # TODO: Rewrite to match refactoring in utilicalcMove - does not refer to spice currently
    def utilicalcSpice(self, agent, pMove):
        circumstances = {}
        agentX, agentY = agent.x, agent.y

        circumstances["intensity"] = agent.getWelfare(x=pMove[0],
                                                      y=pMove[1])  # based off of how much they need sugar/spice
        circumstances["duration"] = 1  # TODO: change to delta time to live
        circumstances["certainty"] = 1 if getDistance(agentX, agentY, pMove[0], pMove[
            1]) <= agent.getVision() else 0  # certainty is their distance from the food. 0 if they cannot see the food.
        return circumstances

    def utilicalcDisease(self, agent, pMove):
        circumstances = {}
        agentX, agentY = agent.x, agent.y

        if agent != self:
            # if agents are nearby the potential move location, we must account for potential sickness
            if getDistance(agentX, agentY, pMove[0], pMove[1]) == 1 and self.isDiseased():
                circumstances["fecundity"] = 1
                circumstances["f_intensity"] = 1 / agent.getSugarMetabolism()
                circumstances["f_duration"] = self.env.getDiseaseLength()
                circumstances["f_proximity"] = 1
                circumstances["f_extent"] = 1

        else:
            numDiseasedNearby = 0
            foodNeighbors = self.env.getNeighborhood(pMove[0], pMove[1])
            for foodNeighbor in foodNeighbors:
                if foodNeighbor != self and \
                        (foodNeighbor.x == agent.x or foodNeighbor.y == agent.y) and \
                        getDistance(foodNeighbor.x, foodNeighbor.y, agent.x, agent.y) <= agent.getVision():
                    numDiseasedNearby += foodNeighbor.isDiseased()

            circumstances["purity"] = 1
            circumstances["p_intensity"] = numDiseasedNearby / agent.getSugarMetabolism()  # if none are diseased, 0
            circumstances["p_duration"] = self.env.getDiseaseLength()
            circumstances["p_proximity"] = 1
            circumstances["p_extent"] = 1
        return circumstances

    def utilicalcSugar(self, agent, pMove):
        circumstances = {}
        agentX, agentY = agent.x, agent.y
        siteWealth = self.env.getSugarAmt(pMove)
        daysToDeath = agent.getDaysToStarvation()

        circumstances["intensity"] = 1 / (1 + daysToDeath)
        circumstances["duration"] = (siteWealth / agent.sugarMetabolism) / self.env.getMaxCapacity()
        circumstances["certainty"] = 1 if getDistance(agentX, agentY, pMove[0], pMove[1]) <= agent.getVision() and (agent.x == pMove[0] or agent.y == pMove[1]) else 0  # certainty is their distance from the food. 0 if they cannot see the food.
        circumstances["proximity"] = 1
        circumstances["futureBliss"] = 0
        # circumstances["fecundity"] = 0
        return circumstances

    '''
    TODO: rework utilicalc movement functions to this form
    for move in potentialMoves:
        # repeat next two lines for every option/toggle (e.g. combat, spice, trade)
        for agent in visionNeighborhood:
            compute utilicalc circumstances - dictionary key: (agent, move), value: [circumstances]
        if self does move:
            compute - dictionary key: move, value: visionNeighborhood's utility --> moveImpacts
    return max(moveImpacts)
    
    We have concluded that intensity, duration, and certainty are correct. We need to normalize/scale them to either TTL or total resource value.
    Certainty will change once we implement a separation of movement and vision.
    Currently proximity is 1, because if you can see it, you can eat it. Again, this will change when we change movement from vision. (1/distance in time steps)
    Fecundity/purity are a single magical variable named futureBliss. The futureBliss variable represents the probability of future pleasure.
    Extent is # of agents in our vision.
    We will assume omniscience within our own neighborhood.
    
    utilicalcSugar:
    a.	Intensity: [0 : 1] (1/1+daysToDeath)
    b.	Duration: [0 : 1] [(cell site wealth / agent metabolism) / maxSiteWealth], which is rational
    c.  Certainty: [1 : 1] if agent can reach move cell, [0 : 0] otherwise
    d.  Proximity: [0 : 1] the (1/distance in time steps), which is currently 1
    e.  FutureBliss: [0 : 1] probability of immediate (or limited by computational horizon) future pleasure subsequent to this action
    f.  Extent: (0 : 1] number of agents in neighborhood  / #agents_visible_in_maxVision
    
    utility_of_cell = certainty * proximity * (intensity + duration + discount * futureBliss * futureReward??? + extent)
    We may wish to weight the variables inside the parenthesis based on some relative importance that we will make up, based on how we think Bentham thought.
    And of course, we will be right.
    '''

    # alternative to move function - moves using felicific_calculus.py functions for moral decision making
    def utilicalcMove(self):

        self.previousWealth = self.getWealth()

        # build a list of available food locations
        potentialMoves = self.getFood() if not self.env.getHasCombat() else self.getFoodWithCombat(self.getPreys())

        # randomize food locations
        random.shuffle(potentialMoves)

        potentialMoves.append((self.x, self.y))

        # list of agents
        visionNeighborhood = self.getVisionNeighbourhood()
        visionNeighborhood.append(self)

        # if self.env.getSelfInterestScale() is not None:
        #    actions.setSelfInterestScale(self.env.getSelfInterestScale())

        circumstances = {}
        baseWeight = 1
        diseaseWeight = .3
        pollutionWeight = .2
        secondOrderDiscount = .5
        for move in potentialMoves:
            for agent in visionNeighborhood:
                key = str(agent.id) + " " + str(move)
                if self.env.getHasCombat():
                    combatCircumstances = self.utilicalcCombat(agent, move)
                    for circumstanceType in combatCircumstances.keys():
                        combatCircumstances[circumstanceType] *= baseWeight
                    circumstances[key] = combatCircumstances
                    circumstances[key]["extent"] = len(visionNeighborhood) - 1

                elif self.env.getHasSpice():
                    spiceCircumstances = self.utilicalcSpice(agent, move)
                    for circumstanceType in spiceCircumstances:
                        spiceCircumstances[circumstanceType] *= baseWeight
                    circumstances[key] = spiceCircumstances
                    circumstances[key]["extent"] = len(visionNeighborhood) - 1

                else:
                    sugarCircumstances = self.utilicalcSugar(agent, move)
                    for circumstanceType in sugarCircumstances:
                        sugarCircumstances[circumstanceType] *= baseWeight
                    circumstances[key] = sugarCircumstances
                    circumstances[key]["extent"] = len(visionNeighborhood) - 1

                if self.env.getHasDisease():
                    diseaseCircumstances = self.utilicalcDisease(agent, move)
                    for circumstanceType in diseaseCircumstances:
                        if circumstanceType in circumstances[key]:
                            circumstances[key][circumstanceType] += diseaseWeight * diseaseCircumstances[circumstanceType]
                        else:
                            circumstances[key][circumstanceType] = diseaseWeight * diseaseCircumstances[circumstanceType]

                if self.env.getHasPollution():
                    pollutionCircumstances = self.utilicalcPollution(agent, move)
                    for circumstanceType in pollutionCircumstances:
                        if circumstanceType in circumstances[key]:
                            circumstances[key][circumstanceType] += pollutionWeight * pollutionCircumstances[circumstanceType]
                        else:
                            circumstances[key][circumstanceType] = pollutionWeight * pollutionCircumstances[circumstanceType]

        # we now have circumstances for every location in vision and agent in vision at each location and need to
        # calculate total cell utilities for each move. utility_of_cell = certainty * proximity * (intensity +
        # duration + discount * futureBliss * futureReward??? + extent)

        cellUtilities = {}
        for move in potentialMoves:
            cellUtilities[move] = 0
            for agent in visionNeighborhood:
                if str(agent.id) + " " + str(move) in circumstances:
                    locCirc = circumstances[str(agent.id) + " " + str(move)]  # location circumstances
                    agentUtility = 0
                    agentUtility += locCirc["certainty"] * locCirc["proximity"] * (
                                locCirc["intensity"] + locCirc["duration"] + secondOrderDiscount * locCirc["futureBliss"] + locCirc["extent"])
                    agentUtility *= -1 if agent != self else 1
                    cellUtilities[move] += agentUtility

        max_location = max(cellUtilities, key=cellUtilities.get)

        bestMove = max_location
        newx, newy = bestMove

        killed = None  # combat
        if self.env.getHasCombat():
            killed = self.env.getAgent((newx, newy))
            if killed is not None:
                self.addLogEntry("I killed Agent" + str(killed.getId) + "at" + str(killed.getLocation()))
                self.setSugar(min(self.sugar + killed.sugar, self.sugar + self.env.getCombatAlpha()),
                              "kill of agent " + str(killed.getId))

        self.env.setAgent((self.x, self.y), None)
        self.env.setAgent((newx, newy), self)
        previousLocation = (self.x, self.y)
        self.x = newx
        self.y = newy
        self.addLogEntry(str("move: " + str(previousLocation) + " -> " + str((newx, newy))))

        best_sugar = self.env.getSugarAmt((newx, newy))
        best_spice = self.env.getSpiceAmt((newx, newy))
        self.setSugar(max(self.sugar + best_sugar, 0), "Move")
        if self.env.getHasSpice():
            self.setSpice(max(self.spice + best_spice, 0), "Move")

        self.env.setSugarAmt((self.x, self.y), 0)
        self.env.setSpiceAmt((self.x, self.y), 0)

        if self.env.getHasPollution():
            self.env.polluteSite((newx, newy), best_sugar, 0)

        if self.env.getHasCombat():
            return killed

    """
    Concept: Trade

    Follows Agent trade rule T from Growing Artificial Societies by Epstein and Axtell, Pg. 105

    "Agent and neighbor compute their MRSs; if these are equal then end, else continue

    the direction of exchange is as follows: spice follows from the agent with the higher MRS to the agent with the 
    lower MRS while sugar goes in the opposite direction

    the geometric mean of the two MRSs is calculated--this will serve as the price, p;

    the quantities to be exchanged are as follows: if p > 1 then p units of spice for 1 unit of sugar; if p < 1 then
    1/p units of sugar for 1 unit of spice

    if this trade will make both agents better off (increases the welfare of both agents), and not cause the agents 
    MRSs to cross over one another, then the trade is made and return to start, else end"
    """

    def trade(self):
        # Formula starts on page 102 of the book "Growing Artificial Societies" by Epstein and Axtell
        # in this scenario, self is A and other is B
        trades = []
        potential_traders = self.getNeighbourhood()
        random.shuffle(potential_traders)
        for trader in potential_traders:
            if trader is not None:
                trader_1 = self
                trader_2 = trader

                has_traded = True  # if no trade is made, this will be false and the loop will break, otherwise it will continue
                while has_traded:
                    trader_1_mrs = self.getMRS()
                    trader_2_mrs = trader.getMRS()
                    trader_1_welfare = self.getWelfare()
                    trader_2_welfare = trader.getWelfare()
                    p = math.sqrt(trader_1_mrs * trader_2_mrs)

                    if trader_1_mrs > trader_2_mrs:
                        has_traded = tradeHelper(trader_1, trader_2, p, trader_1_welfare, trader_2_welfare)

                    elif trader_1_mrs < trader_2_mrs:
                        has_traded = tradeHelper(trader_2, trader_1, p, trader_2_welfare, trader_1_welfare)

                    else:
                        has_traded = False

                    if has_traded:
                        trades.append(p)
        return trades

    # MRS means Marginal Rate of Substitution
    def getMRS(self, w1=None, w2=None):
        m1 = self.sugarMetabolism
        m2 = self.spiceMetabolism

        if not w1 and not w2:
            w1 = self.sugar
            w2 = self.spice

        t1 = w1 / m1
        t2 = w2 / m2

        return t2 / t1 if t1 != 0 else 0

    """
    Concept: Welfare

    Follows Agent welfare function from Growing Artificial Societies by Epstein and Axtell, Pg 97

    "One way to compute this (relative need of sugar or spice) is to have the agents compute how "close" they are to starving 
    to death due to a lack of either sugar or spice. They then attempt to gather relatively more of the good whose absence 
    most jeopardizes their survival"

    This is extended further to be a useful function in computing trade and best location to move to.
    """

    def getWelfare(self, w1=None, w2=None, x=None, y=None):
        m1 = self.sugarMetabolism
        m2 = self.spiceMetabolism

        if not w1 and not w2:
            w1 = self.sugar
            w2 = self.spice

        if x and y:
            x1 = self.env.getSugarAmt((x, y))
            x2 = self.env.getSpiceAmt((x, y))

            w1 = self.sugar + x1
            w2 = self.spice + x2

        mt = m1 + m2

        if self.env.getHasTags():
            # follows from page 125 of the book "Growing Artificial Societies" by Epstein and Axtell
            tags = self.getTags()
            f = float(bin(tags).count('1')) / float(self.tagsLength)
            u = m1 * f + m2 * (1 - f)
            return w1 ** ((m1 / u) * f) * w2 ** ((m2 / u) * (1 - f)) if w1 > 0 and w2 > 0 else 0

        if self.env.getHasForesight() and x and y:
            # follows from page 129 and 130 of the book "Growing Artificial Societies" by Epstein and Axtell
            sugarForesight = (w1 - self.foresight * m1) if w1 - self.foresight * m1 > 0 else 0
            spiceForesight = (w2 - self.foresight * m2) if w2 - self.foresight * m2 > 0 else 0

            return sugarForesight ** (m1 / mt) * spiceForesight ** (m2 / mt) if w1 > 0 and w2 > 0 else 0

        return w1 ** (m1 / mt) * w2 ** (m2 / mt) if w1 > 0 and w2 > 0 else 0

    """
    Concept: Combat
    
    Follows Agent combat rule C_a from Growing Artificial Societies by Epstein and Axtell, Pg. 83

    "Look out as far as vision permits in the four principal lattice directions

    Throw out all sites occupied by members of the agent's own tribe

    Throw out all sites occupied by members of different tribes who are wealthier than the agent

    The reward of each remaining site is given by the resource level at the site plus, if it is occupied, the minimum of 
    a and the occupant's wealth

    Throw out all sites that are vulnerable to retaliation

    Select the nearest position having maximum reward and go there

    Gather the resources at the site plus the minimum of a and the occupant's wealth, if the site was occupied

    If the site was occupied, then the former occupant is considered "killed" -- permanently removed from play"
    """

    def combat(self, alpha):
        food = self.getFoodWithCombat(self.getPreys())

        # randomize food locations
        random.shuffle(food)

        # find best food location
        move = False
        newx = self.x
        newy = self.y
        best = self.env.getSugarAmt((self.x, self.y))
        minDistance = 0
        for (x, y) in food:
            capacity = self.env.getSugarAmt((x, y))
            agent = self.env.getAgent((x, y))
            if agent:
                capacity += min(alpha, agent.getSugar())
            distance = abs(x - self.x + y - self.y)  # Manhattan distance enough due to no diagonal
            if capacity > best or (capacity == best and distance < minDistance):
                best = capacity
                minDistance = distance
                move = True
                newx = x
                newy = y

        # move to new location if any, and kill if occupied
        killed = None
        if move:
            killed = self.env.getAgent((newx, newy))
            self.env.setAgent((self.x, self.y), None)
            self.env.setAgent((newx, newy), self)
            self.x = newx
            self.y = newy

        if killed is not None:
            self.addLogEntry("I killed Agent" + str(killed.getId) + "at" + str(killed.getLocation()))

        # collect, eat and consume
        self.setSugar(max(self.sugar + best, 0), "Move")
        self.env.setSugarAmt((self.x, self.y), 0)
        return killed

    """
    Concept: Disease

    Follows agent immune response rule and agent disease transmission rule from pgs. 144-145 of the book "Growing Artificial Societies" by Epstein and Axtell

    The immune response rule is as follows:
    "If the disease is a substring of the immune system then end (the agent is immune), 
    else (the agent is infected) go to the following step:

    The substring in the agent immune system having the smallest Hamming distance from the disease is selected and 
    the first bit at which it is different from the disease string is changed to match the disease."

    The disease transmission rule is as follows:
    "For each neighbor, a disease that currently afflicts the agent is selected at random and given to the neighbor."
    """

    def getHammingDistance(self,
                           disease):  # returns the smallest bitwise distance between a substring of agent immune system and disease
        diseaseLength = self.env.getDiseaseLength()
        lowestNumber = sys.maxsize
        bestLoc = 0
        for i in range(len(self.immuneSystem) - diseaseLength):
            immuneSystemSubstr = self.immuneSystem[i:i + diseaseLength]
            smallestDist = 0
            for j in range(len(immuneSystemSubstr)):
                smallestDist += 1 if immuneSystemSubstr[j] != disease[j] else 0
            if smallestDist < lowestNumber:
                bestLoc = i
                lowestNumber = smallestDist
        return lowestNumber, bestLoc  # if lowestNumber is 0, agent is immune

    def addDisease(self, disease):
        smallestDist, bestLoc = self.getHammingDistance(disease)
        if smallestDist != 0:
            self.diseases[disease] = bestLoc
            self.sugarMetabolism += 1
            self.addLogEntry("Contracted disease: " + str(disease))

    def updateImmuneSystem(self):
        diseaseLength = self.env.getDiseaseLength()

        for disease, loc in list(self.diseases.items()):
            currentSubstr = self.immuneSystem[loc:loc + diseaseLength]

            if currentSubstr == disease:
                self.removeDisease(disease)
                continue

            newSubstr = updateHelper(disease, currentSubstr)
            self.immuneSystem = self.immuneSystem.replace(currentSubstr, newSubstr, 1)
            hammingDist, loc = self.getHammingDistance(disease)
            self.diseases[disease] = loc

            if hammingDist == 0:
                self.removeDisease(disease)
                continue

    def removeDisease(self, disease):
        del self.diseases[disease]
        self.sugarMetabolism -= 1
        self.addLogEntry("Gained immunity to disease: " + str(disease))

    def disease(self):  # give random disease to each neighbor
        self.updateImmuneSystem()
        if len(self.diseases) > 0:
            for neighbor in self.getNeighbourhood():
                neighbor.addDisease(random.choice(list(self.diseases.keys())))

    def addRandomDisease(self):
        self.addDisease(random.choice(self.env.getDiseases()))

    def createImmuneSystem(self):
        for i in range(self.env.getImmuneSystemSize()):
            self.immuneSystem += str(random.randint(0, 1))
        self.setUntrainedImmuneSystem()
