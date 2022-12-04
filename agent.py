'''
Created on 2010-04-21

@author: rv
'''
import math
import random
from itertools import product


class Agent:
    '''
    classdocs
    '''

    tagsProbability = 0.55

    def __init__(self, env, x=0, y=0, sugarMetabolism=4, spiceMetabolism=4, vision=6, sugarEndowment=25,
                 spiceEndowment=25, maxAge=100, sex=0, fertility=(12, 50), tags=(0, 11)):
        '''
        Constructor
        '''
        self.env = env
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
        self.sugarCostForMatingRange = 0, 0  # 10,40
        self.spiceCostForMatingRange = 0, 0
        self.tradeLog = []
        self.sugarLog = []
        self.spiceLog = []
        self.log = []
        self.setInitialSugarEndowment(sugarEndowment)
        self.setInitialSpiceEndowment(spiceEndowment)
        self.setTags(tags)

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
        return (self.x, self.y)

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

    def setSugar(self, amount, reason=""):
        if amount != 0:
            previousSugar = self.sugar
            self.sugar = amount
            sugarLog_str = "sugar: " + str(previousSugar) + " -> " + str(self.sugar) + " (" + reason + ")"
            self.sugarLog.append(sugarLog_str)
            self.addLogEntry(sugarLog_str)

    def setSpice(self, amount, reason=""):
        if amount != 0:
            previousSpice = self.spice
            self.spice = amount
            spiceLog_str = "spice: " + str(previousSpice) + " -> " + str(self.spice) + " (" + reason + ")"
            self.spiceLog.append(spiceLog_str)
            self.addLogEntry(spiceLog_str)

    def getSpice(self):
        return self.spice

    def setAge(self, maxAge):
        self.maxAge = maxAge
        self.age = 0

    def getAge(self):
        return self.age

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

    ''' 
    build common lists
    '''

    # build a list of available food locations
    def getFood(self):
        food = [(x, self.y) for x in range(self.x - self.vision, self.x + self.vision + 2)
                if self.env.isLocationValid((x, self.y))
                and self.env.isLocationFree((x, self.y))]

        food.extend([(self.x, y) for y in range(self.y - self.vision, self.y + self.vision + 2)
                     if self.env.isLocationValid((self.x, y))
                     and self.env.isLocationFree((self.x, y))])
        return food

    # build a list of possible neighbours for in neighbourhood
    def getNeighbourhood(self):
        neighbourhood = [self.env.getAgent((x, self.y)) for x in range(self.x - 1, self.x + 2)
                         if self.env.isLocationValid((x, self.y))
                         and not self.env.isLocationFree((x, self.y))
                         and x != self.x]

        neighbourhood.extend([self.env.getAgent((self.x, y)) for y in range(self.y - 1, self.y + 2)
                              if self.env.isLocationValid((self.x, y))
                              and not self.env.isLocationFree((self.x, y))
                              and y != self.y])
        return neighbourhood

    # build a list of possible preys around
    def getPreys(self):
        preys = [self.env.getAgent((x, self.y)) for x in range(self.x - self.vision, self.x + self.vision + 2)
                 if self.env.isLocationValid((x, self.y))
                 and not self.env.isLocationFree((x, self.y))
                 and self.sugar > self.env.getAgent((x, self.y)).getSugar()
                 and self.env.getAgent((x, self.y)).getTribe() != self.tribe]

        preys.extend([self.env.getAgent((self.x, y)) for y in range(self.y - self.vision, self.y + self.vision + 2)
                      if self.env.isLocationValid((self.x, y))
                      and not self.env.isLocationFree((self.x, y))
                      and self.sugar > self.env.getAgent((self.x, y)).getSugar()
                      and self.env.getAgent((self.x, y)).getTribe() != self.tribe])
        return preys

    ''' 
    rules
    '''

    # TRANSMIT
    def transmit(self):
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

    # AGEING
    # If age > maxAge Then the agent is dead (return False)
    def incAge(self):
        self.age += 1
        self.addLogEntry("age: " + str(self.age))
        return max(self.maxAge - self.age, 0)

    # FERTILITY
    # First, to have offspring, agents must be of childbearing age.
    # Second, children born with literally no initial endowment of sugar would instantly die.
    # We therefore require that parents give their children some initial endowment.
    # Each newborn's endowment is the sum of the (usually unequal) contributions of mother and father.
    # Dad contributes an amount equal to one half of whatever his initial endowment had been, and likewise for mom.
    # To be parents, agents must have amassed at least the amount of sugar which they were endowed at birth.
    def isFertile(self):
        if self.fertility[0] <= self.age <= self.fertility[1]:
            if self.sugar >= self.spiceCostForMating and not self.env.getHasSpice():
                return True
            elif self.sugar >= self.sugarCostForMating and self.env.getHasSpice():
                if self.spice >= self.spiceCostForMating:
                    return True
        return False

    # MATE (Generator):
    # Select a neighboring agent at random.
    # If the neighbor is fertile and of the opposite sex and at least one of the agent has an empty neighboring site, then a child is born
    # Repeat for all neighbors.
    def mate(self):
        # build a list of possible partners in neighbourhood
        neighbourhood = self.getNeighbourhood()

        # randomize
        random.shuffle(neighbourhood)

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

        genitors[0].setSpice(max(genitors[0].spice - 0.5 * genitors[0].spiceEndowment, 0), "Create child")
        genitors[1].setSpice(max(genitors[1].spice - 0.5 * genitors[1].spiceEndowment, 0), "Create child")

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
        self.env.setAgent((x, y), child)
        return child

    def getLocationWelfare(self, x, y):
        # agent welfare function to determine if we need to find spice or sugar
        # this formula is based on page 97 of the book "Growing Artificial Societies" by Epstein and Axtell
        m1 = self.sugarMetabolism
        m2 = self.spiceMetabolism

        x1 = self.env.getSugarCapacity((x, y))
        x2 = self.env.getSpiceCapacity((x, y))

        w1 = self.sugar + x1
        w2 = self.spice + x2

        mt = m1 + m2

        return w1 ** (m1 / mt) * w2 ** (m2 / mt)

    def getManhattanDistance(self, x, y):
        return abs(self.x - x) + abs(self.y - y)

    # MOVE:
    # Look out as far as vision permits in the four principal lattice directions and identify the unoccupied site(s) having the most sugar.
    # If the greatest sugar value appears on multiple sites then select the nearest one.
    # Move to this site.
    # Collect all the sugar at this new position.
    # Increment the agent's accumulated sugar wealth by the sugar collected and decrement by the agent's metabolic rate.

    def move(self):
        # find best food location
        move = False
        newx = self.x
        newy = self.y

        # build a list of available food locations
        food = self.getFood()

        # randomize food locations
        random.shuffle(food)
        locations = []
        food.append((self.x, self.y))

        if not self.env.getHasSpice():
            for (x, y) in food:
                location = (x, y)
                sugarCapacity = self.env.getCapacity((x, y))
                distance = self.getManhattanDistance(x, y)  # Manhattan distance enough due to no diagonal
                locations.append((location, sugarCapacity, distance))

            locations.sort(key=lambda x: x[2])
            best_location = max(locations, key=lambda x: x[1])

            if best_location[0] != (self.x, self.y):
                move = True
                newx, newy = best_location[0]

            best_sugar = best_location[1]
            self.setSugar(max(self.sugar + best_sugar, 0), "Move")


        else:
            for (x, y) in food:
                welfare = self.getLocationWelfare(x, y)
                location = (x, y)
                sugarCapacity = self.env.getSugarCapacity((x, y))
                spiceCapacity = self.env.getSpiceCapacity((x, y))
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

        self.env.setSugarCapacity((self.x, self.y), 0)
        self.env.setSpiceCapacity((self.x, self.y), 0)

    def getWelfare(self, w1=None, w2=None):
        m1 = self.sugarMetabolism
        m2 = self.spiceMetabolism

        if not w1 and not w2:
            w1 = self.sugar
            w2 = self.spice

        mt = m1 + m2

        return w1 ** (m1 / mt) * w2 ** (
                    m2 / mt) if w1 > 0 and w2 > 0 else 0

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

    def tradeHelper(self, A, B, p, A_welfare, B_welfare):
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
                        has_traded = self.tradeHelper(trader_1, trader_2, p, trader_1_welfare, trader_2_welfare)

                    elif trader_1_mrs < trader_2_mrs:
                        has_traded = self.tradeHelper(trader_2, trader_1, p, trader_2_welfare, trader_1_welfare)

                    else:
                        has_traded = False

                    if has_traded:
                        trades.append(p)
        return trades

    # TODO: Implement spice with combat?
    # COMBAT
    def combat(self, alpha):
        hasSpice = self.env.getHasSpice()

        # build a list of available unoccupied food locations
        food = self.getFood()

        # build a list of potential preys
        preys = self.getPreys()

        # append to food safe preys (retaliation condition)
        C0 = self.sugar - self.sugarMetabolism
        if not hasSpice:
            food.extend([preyA.getLocation() for preyA, preyB in product(preys, preys)
                         if preyA != preyB
                         and preyB.getSugar() < (
                                     C0 + self.env.getCapacity(preyA.getLocation()) + min(alpha, preyA.getSugar()))])
        else:
            C1 = self.spice - self.spiceMetabolism
            food.extend([preyA.getLocation() for preyA, preyB in product(preys, preys)
                         if preyA != preyB
                         and preyB.getSugar() < (
                                     C0 + self.env.getCapacity(preyA.getLocation()) + min(alpha, preyA.getSugar()))
                         and preyB.getSpice() < (
                                     C1 + self.env.getCapacity(preyA.getLocation()) + min(alpha, preyA.getSpice()))])

        # randomize food locations
        random.shuffle(food)

        # find best food location
        move = False
        newx = self.x
        newy = self.y
        best = self.env.getCapacity((self.x, self.y))
        minDistance = 0
        for (x, y) in food:
            capacity = self.env.getCapacity((x, y))
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
            self.addLogEntry("I killed Agent at" + str(killed.getLocation()))

        # collect, eat and consume
        self.setSugar(max(self.sugar + best, 0), "Move")
        if hasSpice:
            self.setSpice(max(self.spice + best, 0), "Move")
        self.env.setCapacity((self.x, self.y), 0)
        return killed
