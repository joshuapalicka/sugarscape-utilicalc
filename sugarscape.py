'''
Created on 2010-04-17

@author: rv

Updated/extended by joshuapalicka
'''
import math
import random
import time
from itertools import product
from environment import Environment
from agent import Agent
import tkinter as tk

import matplotlib.pyplot as plt
import matplotlib

''' 
initial simulation parameters
'''

# view
screenSize = 600, 600
gridSize = 50, 50
colorBackground = 250, 250, 250
graphUpdateFrequency = 5  # graphs update every graphUpdateFrequency frames

# display colors
colors = {
    "sugar": "#F2FA00",
    "spice": "#9B4722",
    "both": "#BF8232",
    "red": "#FA3232",
    "pink": "#FA32FA",
    "blue": "#3232FA",
    "pollution": "#88C641"
}

# environment
sites = {
    "northeast": (35, 15, 18),
    "southwest": (15, 35, 18),
    "southeast": (40, 40, 14),
    "northwest": (10, 10, 14)
}

maxCapacity = 10
seasonPeriod = 50

seasonRegions = {
    "north": (0, 0, 49, 24),
    "south": (0, 25, 49, 49)
}

growFactor = 1
growFactor1 = 1
growFactor2 = float(growFactor1) / 8

# agents
# agentColorScheme: Agents colour meaning = 0:all, 1:bySex, 2:byMetabolism, 3:byVision, 4:byGroup
maxAgentMetabolism = 5
maxAgentVision = 5
initEndowment = 50, 100
minmaxAgentAge = 60, 100

female = 0
male = 1

fertility = [(12, 15, 40, 50),
             (12, 15, 50, 60),
             (12, 15, 30, 40),
             (12, 15, 40, 50)]
childbearing = fertility[0], fertility[1]  # female , male

tagsLength = 5  # must be odd
tags0 = 0
tags1 = 2 ** tagsLength - 1

loanRate = 1.05
loanDuration = 10

# There exists 2^(diseaseLength) unique diseases and 2^immuneSystemSize unique immune systems
immuneSystemSize = 10  # number of bits per immune system
diseaseLength = 3  # number of bits per disease
numDiseases = 5
numStartingDiseases = 2

rules = {
    "grow": True,
    "seasons": False,
    "moveEat": True,  # move eat and combat need to be exclusive
    "canStarve": True,
    "pollution": False,
    "tags": True,
    "combat": False,
    "limitedLifespan": True,
    "replacement": False,
    "procreate": True,
    "transmit": False,
    "spice": True,  # following must be off for pollution
    "trade": True,
    "foresight": False,
    "credit": False,
    "inheritance": False,
    "disease": False
}

if rules["tags"]:
    agentColorScheme = 4

else:
    agentColorScheme = 1

if rules["pollution"]:
    pA = .1  # production pollution
    pB = .1  # consumption pollution
    pDiffusionRate = 5
    pollutionStartTime = 50
    diffusionStartTime = 100

if rules["foresight"]:
    foresightRange = (0, 10)

distributions = [
    # if rules["tags"] == True then these distributions will be used fully, otherwise, just the first item from each tuple will be used as the number of total agents
    (125, tags0, (0, 50, 0, 50)),  # blues
    (125, tags1, (0, 50, 0, 50))]  # reds

boundGraphData = True
numDeviations = 2  # set graph bounds to be within numDeviations standard deviations of the mean if boundGraphs is True

isRandom = True
combatAlpha = 1000000

if not isRandom:
    random.seed(0)

''' 
Global functions
'''


def hexToRGB(colorHex):
    """ #FFFFFF -> (255, 255, 255) """
    colorHex = colorHex.lstrip('#')
    colorHexLen = len(colorHex)
    return tuple(int(colorHex[i:i + colorHexLen // 3], 16) for i in range(0, colorHexLen, colorHexLen // 3))


def RGBToHex(rgb):
    """ (255, 255, 255) -> #FFFFFF """
    return '#%02x%02x%02x' % rgb


def lightenColorByCapacity(color, amountAtLocation):
    """ lighten color by factor """
    factor = amountAtLocation / maxCapacity
    rgb = hexToRGB(color)
    return RGBToHex(tuple(int(c + (255 - c) * (1 - factor)) for c in rgb))


def lightenColorByX(color, x, maxX):
    factor = x / maxX
    rgb = hexToRGB(color)
    return RGBToHex(tuple(int(c + (255 - c) * (1 - factor if factor < 1 else 0)) for c in rgb))


def initAgent(agent, tags, distribution):
    if rules["tags"]:
        newLocation = agent.getEnv().getRandomFreeLocation(distribution)
    else:
        newLocation = agent.getEnv().getRandomFreeLocation((0, gridSize[0] - 1, 0, gridSize[1] - 1))
    if newLocation is None:
        return False
    agent.setLocation(newLocation)
    agent.setSugarMetabolism(random.randint(1, maxAgentMetabolism))
    agent.setInitialSugarEndowment(random.randint(initEndowment[0], initEndowment[1]))

    if rules["spice"]:
        agent.setSpiceMetabolism(random.randint(1, maxAgentMetabolism))
        agent.setInitialSpiceEndowment(random.randint(initEndowment[0], initEndowment[1]))
    agent.setVision(random.randint(1, maxAgentVision))
    agent.setAge(random.randint(minmaxAgentAge[0], minmaxAgentAge[1]))
    sex = random.randint(0, 1)
    agent.setSex(sex)
    agent.setFertility((random.randint(childbearing[sex][0], childbearing[sex][1]),
                        random.randint(childbearing[sex][2], childbearing[sex][3])))
    if tags is None:
        tags = random.getrandbits(tagsLength)
    agent.setTags((tags, tagsLength))
    return True


def calculateGini(wealth):
    """ Calculate the Gini coefficient of a list of wealth values """
    # based on https://planspace.org/2013/06/21/how-to-calculate-gini-coefficient-from-raw-data-in-python/

    wealth = sorted(wealth)
    height, area = 0, 0
    for value in wealth:
        height += value
        area += height - value / 2
    fair_area = height * len(wealth) / 2
    return (fair_area - area) / fair_area


def getStandardDeviation(data):
    mean = sum(data) / len(data)
    variance = sum([(x - mean) ** 2 for x in data]) / len(data)
    return math.sqrt(variance)


def boundData(data, index):
    if boundGraphData:
        if len(data) > 0:
            std = getStandardDeviation(data)
            boundedData = []
            modifiedIndex = []
            mean = sum(data) / len(data)
            bounds = std * numDeviations
            for i in range(len(index)):
                if mean - bounds < data[i] < mean + bounds:
                    boundedData.append(data[i])
                    modifiedIndex.append(index[i])
            return boundedData, modifiedIndex
    return data


''' 
View Class
'''


class View:

    # this gets called first
    def __init__(self, screenSize, env, agents):
        # init view
        self.stats = {}
        self.wealthWidget, self.metabolismWidget, self.popWidget = None, None, None
        self.mainWindow, self.canvas = None, None
        self.height, self.width = screenSize[0] + 50, screenSize[1]
        self.pause = False
        self.updateScreen = True
        self.quit = False
        self.siteSize = screenSize[0] / env.gridWidth
        self.radius = int(self.siteSize * 0.5)
        self.colorByPollution = False
        self.agentColorScheme = agentColorScheme
        # init env
        self.env = env
        self.season = ""
        # init agents population
        self.agents = agents
        self.population = []

        self.metabolismMean = []
        self.visionMean = []
        self.percentBlueTags = []
        self.tradePriceMean = []
        self.tradeVolumeMean = []
        self.gini = []
        self.foresightMean = []
        self.numInfectedAgents = []
        self.proportionInfectedAgents = []

        self.statsWindow = None

        # init time
        self.iteration = 0
        self.grid = [[(None, None) for __ in range(env.gridWidth)] for __ in range(env.gridHeight)]

    # display agent switch case (dictionary)
    def all(self, agent):
        return colors["red"]

    def bySex(self, agent):
        if agent.getSex() == female:
            return colors["pink"]
        else:
            return colors["blue"]

    def bySugarMetabolism(self, agent):
        return lightenColorByX(colors["red"], agent.getSugarMetabolism(), maxAgentMetabolism)

    def bySpiceMetabolism(self, agent):
        return lightenColorByX(colors["red"], agent.getSpiceMetabolism(), maxAgentMetabolism)

    def byVision(self, agent):
        return lightenColorByX(colors["red"], agent.getVision(), maxAgentVision)

    def byGroup(self, agent):
        #        if bin(agent.getTags()).count('1') > agent.getTagsLength()>>1:
        if agent.getTribe() == 1:
            return colors["red"]
        else:
            return colors["blue"]

    def byAge(self, agent):
        return lightenColorByX(colors["red"], agent.getAge(), minmaxAgentAge[1])

    def byWealth(self, agent):
        return lightenColorByX(colors["red"], agent.getWealth(), 100)

    def byNumberOfDiseases(self, agent):
        return lightenColorByX(colors["red"], agent.getNumAfflictedDiseases(), numDiseases)

    agentColorSchemes = {0: all, 1: bySex, 2: bySugarMetabolism, 3: byVision, 4: byGroup, 5: byAge, 6: byWealth,
                         7: bySpiceMetabolism, 8: byNumberOfDiseases}

    # remove or replace agent
    def findDistribution(self, tags):
        getTribe = lambda x, y: round(float(bin(x).count('1')) / float(y))
        tribe = getTribe(tags, tagsLength)
        for (n, t, d) in distributions:
            if t is not None and getTribe(t, tagsLength) == tribe:
                # found a distribution for tags
                return d
        else:
            # or return best guess
            return d

    # replace or remove agent
    def removeAgent(self, agent):
        if rules["inheritance"]:
            children = list(agent.getChildren())
            numChildren = len(children)
            if numChildren > 0:
                # inherit sugar and spice
                sugar = agent.getSugar()
                spice = agent.getSpice()
                for child in children:
                    child.setSugar(child.getSugar() + (sugar / numChildren), "inheritance")
                    if rules["spice"]:
                        child.setSpice(child.getSpice() + (spice / numChildren), "inheritance")
                if rules["credit"]:
                    agent.splitLoansAmongChildren(children)

        agent.setAlive(False)

        if rules["replacement"]:
            # replace with agent of same tribe
            tags = agent.getTags()
            newAgent = Agent(env)
            if initAgent(newAgent, tags, self.findDistribution(tags)):
                self.env.setAgent(newAgent.getLocation(), newAgent)
                self.agents.append(newAgent)

        if agent in self.agents:
            self.agents.remove(agent)

    def hasStarved(self, agent):
        if agent.getSugar() <= 0.1:
            agent.addLogEntry("Starved due to lack of sugar")
            # free environment
            agent.setAlive(False)
        if rules["spice"]:
            if agent.getSpice() <= 0.1:
                agent.addLogEntry("Starved due to lack of spice")
                agent.setAlive(False)

    #  put game update code here
    def updateGame(self):
        # for agents' logs
        sugarMetabolism = 0
        if rules["spice"]:
            spiceMetabolism = 0

        vision = 0
        wealth = []
        trades = []

        # execute agents randomly
        random.shuffle(self.agents)

        # run agents' rules
        for agent in self.agents:
            # MOVE
            if rules["moveEat"]:
                agent.move()

            # COMBAT
            if rules["combat"]:
                killed = agent.combat(combatAlpha)
                # if an agent has been killed, remove it
                if killed:
                    # do not free the environment, someone else is already here
                    self.removeAgent(killed)

            if rules["credit"]:
                agent.credit()

            # PROCREATE
            if rules["procreate"] and agent.isFertile():
                mateItr = agent.mate()
                canMate = True
                while canMate:
                    try:
                        # if a new baby is born, append it to the agents' list
                        self.agents.append(next(mateItr))
                    except StopIteration:
                        canMate = False

            if rules["disease"]:
                agent.disease()

            if rules["trade"]:
                trades.extend(agent.trade())

            # TRANSMIT
            if rules["transmit"]:
                agent.transmit()

            if rules["pollution"] and not rules["spice"] and agent.getSugar() > 0:
                self.env.polluteSite((agent.getLocation()), 0, agent.getSugarMetabolism())

            agent.setSugar(max(agent.getSugar() - agent.getSugarMetabolism(), 0), "Metabolism")

            if rules["spice"]:
                agent.setSpice(max(agent.getSpice() - agent.getSpiceMetabolism(), 0), "Metabolism")

            # Log agent's parameters
            sugarMetabolism += agent.getSugarMetabolism()
            if rules["spice"]:
                spiceMetabolism += agent.getSpiceMetabolism()
            vision += agent.getVision()

            if not rules["spice"]:
                wealth.append(agent.getSugar())
            else:
                wealth.append(agent.getSugar() + agent.getSpice())

            if rules["canStarve"]:
                self.hasStarved(agent)

            if agent.isAlive():
                # increment age
                if rules["limitedLifespan"]:
                    agent.setAlive(agent.incAge())

            if not agent.isAlive():
                # free environment
                self.env.setAgent(agent.getLocation(), None)
                # remove or replace agent
                self.removeAgent(agent)

        # Log population
        numAgents = len(self.agents)
        self.population.append(numAgents)

        # Calculate and log agents' metabolism and vision mean values
        if not rules["spice"]:
            self.metabolismMean.append(sugarMetabolism / numAgents if numAgents > 0 else 0)
        else:
            self.metabolismMean.append(
                ((sugarMetabolism + spiceMetabolism) / 2) / float(numAgents) if numAgents > 0 else 0)

        self.visionMean.append(vision / float(numAgents) if numAgents > 0 else 0)

        if rules["trade"]:
            self.tradePriceMean.append(sum(trades) / len(trades)) if len(trades) > 0 else self.tradePriceMean.append(0)
            self.tradeVolumeMean.append(len(trades)) if len(trades) > 0 else self.tradeVolumeMean.append(0)
        self.gini.append(calculateGini(wealth) if numAgents > 0 else 0)

        if rules["foresight"]:
            foresight = 0
            for agent in self.agents:
                foresight += agent.getForesight()
            self.foresightMean.append(foresight / float(numAgents) if numAgents > 0 else 0)

        if rules["tags"]:
            numBlue = 0
            for agent in self.agents:
                if agent.getTribe() == 0:
                    numBlue += 1
            self.percentBlueTags.append((numBlue / float(numAgents)*100) if numAgents > 0 else 0)

        if rules["disease"]:
            numInfected = 0
            for agent in self.agents:
                numInfected += 1 if agent.getNumAfflictedDiseases() > 0 else 0
            self.numInfectedAgents.append(numInfected)
            self.proportionInfectedAgents.append((numInfected / numAgents) if numAgents > 0 else 0)

        # run environment's rules
        if rules["seasons"]:
            S = (self.iteration % (2 * seasonPeriod)) / seasonPeriod
            if S < 1:
                # Summer
                self.season = "(summer, winter)"
                if rules["grow"]:
                    self.env.growRegion(seasonRegions["north"], growFactor1)
                    self.env.growRegion(seasonRegions["south"], growFactor2)
            else:
                # winter
                self.season = "(winter, summer)"
                if rules["grow"]:
                    self.env.growRegion(seasonRegions["north"], growFactor2)
                    self.env.growRegion(seasonRegions["south"], growFactor1)

        elif rules["grow"]:
            self.season = "NA"
            self.env.grow(growFactor)

        if rules["pollution"]:
            if self.iteration >= diffusionStartTime and self.iteration % pDiffusionRate == 0:
                self.env.spreadPollution()

        if self.iteration % graphUpdateFrequency == 0:
            if len(self.onGraphs) > 0:
                self.updateGraphs()

        self.iteration += 1
        env.incrementTime()
        self.updateStatsWindow()

    def getFillColor(self, row, col):
        current_agent = env.getAgent((row, col))

        # change color of site depending on what's on it - but only if it wasn't already that color (performance optimization)
        if current_agent:
            fill_color = self.agentColorSchemes[self.agentColorScheme](self, current_agent)
        elif self.colorByPollution:
            fill_color = lightenColorByX(colors["pollution"], env.getPollutionAtLocation((row, col)), 10)
        else:
            sugarCapacity = env.getSugarCapacity((row, col))
            if not rules["spice"]:
                fill_color = lightenColorByCapacity(colors["sugar"], sugarCapacity)
            else:
                spiceCapacity = env.getSpiceCapacity((row, col))
                if sugarCapacity >= spiceCapacity:
                    fill_color = lightenColorByCapacity(colors["sugar"], sugarCapacity)
                elif sugarCapacity < spiceCapacity:
                    fill_color = lightenColorByCapacity(colors["spice"], spiceCapacity)
                else:
                    fill_color = "white"
        return fill_color

    def draw(self):
        for row, col in product(range(len(self.grid)), range(len(self.grid[0]))):
            fill_color = self.getFillColor(row, col)
            if self.grid[row][col][1] != fill_color:
                self.canvas.itemconfig(self.grid[row][col][0], fill=fill_color)
                self.grid[row][col] = (self.grid[row][col][0], fill_color)

    # put drawing code here
    def initialDraw(self):
        # display Sugarscape
        for row, col in product(range(env.gridHeight), range(env.gridWidth)):
            x1 = 5 + (.5 * self.siteSize) + row * self.siteSize - (.5 * self.siteSize)
            y1 = 5 + (.5 * self.siteSize) + col * self.siteSize - (.5 * self.siteSize)
            x2 = 5 + (.5 * self.siteSize) + row * self.siteSize + (.5 * self.siteSize)
            y2 = 5 + (.5 * self.siteSize) + col * self.siteSize + (.5 * self.siteSize)

            fill_color = self.getFillColor(row, col)
            self.grid[row][col] = (
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline="#C0C0C0"), fill_color)

    def makeStatsDict(self):
        round_to = 2
        self.stats = {
                      "Iteration": self.iteration,
                      "Average Population": round((sum(self.population) / len(self.population)) if len(self.agents) > 0 else 0, round_to),
                      "Metabolism Mean": round(sum(self.metabolismMean) / len(self.metabolismMean), round_to),
                      "Vision Mean": round(sum(self.visionMean) / len(self.visionMean), round_to),
                      "Gini Coefficient": round(self.gini[-1], round_to),
                      "Average Gini": round(sum(self.gini) / len(self.gini), round_to),
                      "Average Wealth": round((sum(self.getAgentsWealth()) / len(self.agents)) if len(self.agents) > 0 else 0, round_to)
                      }

        if rules["trade"]:
            self.stats["Trade Price Mean"] = round(sum(self.tradePriceMean) / len([filter(None, self.tradePriceMean)]), round_to)  # Adding filter(None, ...) to remove 0s from the list
            self.stats["Trade Volume Mean"] = round(sum(self.tradeVolumeMean) / self.iteration, round_to)
        if rules["foresight"]:
            self.stats["Foresight Mean"] = round(sum(self.foresightMean) / len(self.foresightMean), round_to)
        if rules["tags"]:
            self.stats["Percent Blue Tags"] = round(sum(self.percentBlueTags) / len(self.percentBlueTags), round_to)
        if rules["disease"]:
            self.stats["Number of Infected Agents"] = round(sum(self.numInfectedAgents) / len(self.numInfectedAgents), round_to)
            self.stats["Proportion of Infected Agents"] = round(sum(self.proportionInfectedAgents) / len(self.proportionInfectedAgents), round_to)

    def setQuit(self):
        self.quit = True

    def togglePause(self):
        self.pause = not self.pause
        self.btnPlay.config(text="Play Simulation" if self.pause else "Pause Simulation")
        print("Pause: ", self.pause)

    def toggleUpdateScreen(self):
        print("Update: ", self.updateScreen)
        self.updateScreen = not self.updateScreen
        self.btnUpdate.config(text="Play Render" if self.pause else "Pause Render")

    def advanceOneStep(self):
        self.pause = False
        self.step()
        self.pause = True
        self.btnPlay.config(text="Play Simulation")

    def toggleColorByPollution(self):
        self.colorByPollution = not self.colorByPollution
        self.draw()

    def getAgentsWealth(self):
        wealth = []
        for agent in self.agents:
            if not rules["spice"]:
                wealth.append(agent.getSugar())
            else:
                wealth.append(agent.getSugar() + agent.getSpice())
        return wealth

    def updatePopulationPlot(self, ax_idx):
        fig, ax = self.figs[ax_idx]
        ax.clear()
        # create figure
        ax.plot(range(len(self.population)), self.population)
        ax.set_title("Population time series")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Population")
        ax.set_ylim(ymin=0)
        self.figs[ax_idx] = (fig, ax)

    def updateWealthPlot(self, ax_idx):
        fig, ax = self.figs[ax_idx]
        ax.clear()
        # create figure
        ax.hist(self.getAgentsWealth(), bins=10)
        ax.set_title("Wealth histogram")
        ax.set_xlabel("Wealth")
        ax.set_ylabel("Number of agents")
        ax.set_ylim(ymin=0)
        self.figs[ax_idx] = (fig, ax)

    def updateMetabolismVisionPlot(self, ax_idx):
        fig, ax = self.figs[ax_idx]
        ax.clear()
        ax.plot(self.metabolismMean, label="metabolism")
        ax.plot(self.visionMean, label="vision")
        ax.set_title("Agents' metabolism and vision mean values")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mean value")
        ax.set_ylim(ymin=0)
        ax.legend()
        self.figs[ax_idx] = (fig, ax)

    def updateTagProportionPlot(self, ax_idx):
        fig, ax = self.figs[ax_idx]
        ax.clear()
        ax.plot(range(len(self.percentBlueTags)), self.percentBlueTags)
        ax.set_title("Percent Blue tags to Red tags time series")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Percent tags which are Blue")
        ax.set_ylim(ymin=0, ymax=100)
        self.figs[ax_idx] = (fig, ax)

    def updateTradePricePlot(self, ax_idx):
        # create figure
        fig, ax = self.figs[ax_idx]
        ax.clear()
        index = list(range(self.iteration))
        data, index = boundData(self.tradePriceMean, index)
        ax.scatter(x=index, y=data, label="price", s=4)
        ax.set_title("Mean trade price")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mean price")
        ax.legend()
        ax.set_ylim(ymin=0)
        self.figs[ax_idx] = (fig, ax)

    def updateTradeVolumePlot(self, ax_idx):
        # create figure
        fig, ax = self.figs[ax_idx]
        ax.clear()
        ax.plot(self.tradeVolumeMean, label="volume")
        ax.set_title("Mean trade volume")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mean volume")
        ax.set_ylim(ymin=0)
        ax.legend()
        self.figs[ax_idx] = (fig, ax)

    def updateGiniPlot(self, ax_idx):
        # create figure
        fig, ax = self.figs[ax_idx]
        ax.clear()
        ax.plot(self.gini, label="gini")
        ax.set_title("Gini")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Gini")
        ax.set_ylim(ymin=0)
        ax.legend()
        self.figs[ax_idx] = (fig, ax)

    def updateForesightPlot(self, ax_idx):
        fig, ax = self.figs[ax_idx]
        ax.clear()
        # create figure
        ax.plot(range(len(self.foresightMean)), self.foresightMean)
        ax.set_title("Mean Foresight time series")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mean Foresight")
        ax.set_ylim(ymin=0)
        self.figs[ax_idx] = (fig, ax)

    def updateInfectedPlot(self, ax_idx):
        fig, ax = self.figs[ax_idx]
        ax.clear()
        ax.plot(range(len(self.proportionInfectedAgents)), self.proportionInfectedAgents)
        ax.set_title("Proportion of Infected Agents to Healthy Agents time series")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Proportion of Infected Agents")
        ax.set_ylim(ymin=0)
        self.figs[ax_idx] = (fig, ax)

    def populateGraphOptions(self):  # TODO: Add more graphs and options
        # populate graph options depending on the rules
        self.graphOptions = []

        self.graphOptions.append("Population")
        self.graphOptions.append("Wealth")
        self.graphOptions.append("Metabolism and Vision")
        self.graphOptions.append("Gini")

        if rules["tags"]:
            self.graphOptions.append("Tag Proportion")

        if rules["trade"]:
            self.graphOptions.append("Trade Price")
            self.graphOptions.append("Trade Volume")

        if rules["foresight"]:
            self.graphOptions.append("Mean Foresight")

        if rules["disease"]:
            self.graphOptions.append("Proportion Infected")

        self.offGraphs = self.graphOptions.copy()
        self.onGraphs = []

    def populateAgentViewOptions(self):
        self.agentViewOptions = []
        self.agentViewOptions.append("Single Color")
        self.agentViewOptions.append("By Sex")
        self.agentViewOptions.append("By Sugar Metabolism")
        if rules["spice"]:
            self.agentViewOptions.append("By Spice Metabolism")
        self.agentViewOptions.append("By Vision")
        self.agentViewOptions.append("By Group")
        self.agentViewOptions.append("By Age")
        self.agentViewOptions.append("By Wealth")
        if rules["disease"]:
            self.agentViewOptions.append("By Number of Diseases")

    def updateAgentView(self, *args):
        selectedView = self.lastSelectedAgentView.get()
        if selectedView == "Single Color":
            self.agentColorScheme = 0
        elif selectedView == "By Sex":
            self.agentColorScheme = 1
        elif selectedView == "By Sugar Metabolism":
            self.agentColorScheme = 2
        elif selectedView == "By Vision":
            self.agentColorScheme = 3
        elif selectedView == "By Group":
            self.agentColorScheme = 4
        elif selectedView == "By Age":
            self.agentColorScheme = 5
        elif selectedView == "By Wealth":
            self.agentColorScheme = 6
        elif selectedView == "By Spice Metabolism":
            self.agentColorScheme = 7
        elif selectedView == "By Number of Diseases":
            self.agentColorScheme = 8
        self.updateWindow()

    def updateGraphList(self, *args):
        new_onGraph = self.lastSelectedGraph.get()
        idx = self.graphOptions.index(new_onGraph)

        if new_onGraph in self.offGraphs:
            self.offGraphs.remove(new_onGraph)
            self.onGraphs.append(new_onGraph)
            idx = self.graphOptions.index(new_onGraph)
            self.graphMenu.delete(idx)
            self.graphMenu.insert_checkbutton(idx, label=new_onGraph, font='Helvetica 11 bold', onvalue=new_onGraph,
                                              offvalue=new_onGraph,
                                              variable=self.lastSelectedGraph,
                                              command=self.updateGraphList, indicatoron=False)
        elif new_onGraph in self.onGraphs:
            self.onGraphs.remove(new_onGraph)
            self.offGraphs.append(new_onGraph)

            self.graphMenu.delete(idx)
            self.graphMenu.insert_checkbutton(idx, label=new_onGraph, onvalue=new_onGraph, offvalue=new_onGraph,
                                              variable=self.lastSelectedGraph,
                                              command=self.updateGraphList, indicatoron=False)
        self.updateGraphs()

    def handle_close(self, evt):
        fig = self.figs.pop()
        plt.close(fig[0])
        self.onGraphs.remove(fig[1])
        self.offGraphs.append(fig[1])
        # self.updateGraphList()

    def checkAddFig(self):
        while len(self.onGraphs) > len(self.figs):
            fig, ax = plt.subplots()
            self.figs.append((fig, ax))

    def updateGraphs(self):  # once match case becomes more standard, this can be rewritten to be made more pythonic
        self.checkAddFig()
        totalPlots = len(self.onGraphs)
        for i in range(totalPlots):
            if self.onGraphs[i] == "Population":
                self.updatePopulationPlot(i)
            elif self.onGraphs[i] == "Wealth":
                self.updateWealthPlot(i)
            elif self.onGraphs[i] == "Tag Proportion":
                self.updateTagProportionPlot(i)
            elif self.onGraphs[i] == "Metabolism and Vision":
                self.updateMetabolismVisionPlot(i)
            elif self.onGraphs[i] == "Trade Price":
                self.updateTradePricePlot(i)
            elif self.onGraphs[i] == "Trade Volume":
                self.updateTradeVolumePlot(i)
            elif self.onGraphs[i] == "Gini":
                self.updateGiniPlot(i)
            elif self.onGraphs[i] == "Mean Foresight":
                self.updateForesightPlot(i)
            elif self.onGraphs[i] == "Proportion Infected":
                self.updateInfectedPlot(i)
            # plt.gcf().canvas.manager.window.overrideredirect(1)
            self.figs[i][0].canvas.draw()
            self.figs[i][0].canvas.flush_events()

        if totalPlots < len(self.figs):
            for _ in range(len(self.figs) - totalPlots):
                fig = self.figs.pop()
                plt.close(fig[0])

    def on_mainClosing(self):
        self.mainWindow.destroy()
        self.quit = True

    def on_statsClosing(self):
        self.statsWindow.destroy()
        self.statsWindow = None

    def updateStatsWindow(self):
        if self.statsWindow:
            self.makeStatsDict()
            self.text_box.configure(state='normal')
            self.text_box.delete("1.0", tk.END)
            for key, value in self.stats.items():
                self.text_box.insert(tk.END, key + ": " + str(value) + "\n")
            self.text_box.configure(state='disabled')
            self.text_box.pack()

    def createStatsWindow(self):
        if not self.statsWindow:
            self.statsWindow = tk.Tk()
            self.makeStatsDict()
            self.statsWindow.option_add("*font", "Roboto 14")
            self.text_box = tk.Text(self.statsWindow, state='disabled', height=(1.2 * len(self.stats.keys())), width=25)
            self.statsWindow.title("Stats")
            self.statsWindow.protocol("WM_DELETE_WINDOW", self.on_statsClosing)
            self.updateStatsWindow()

        else:
            self.statsWindow.destroy()
            self.statsWindow = None

    # the main game loop
    def createWindow(self):  # TODO: if graph window closed, crash occurs
        self.update = True
        plt.ion()
        self.figs = []
        self.populateGraphOptions()
        self.populateAgentViewOptions()
        self.mainWindow = tk.Tk()
        self.mainWindow.title("Sugarscape")
        self.mainWindow.geometry("%dx%d" % (self.width + 5, self.height - 5))
        self.mainWindow.resizable(True, True)
        self.mainWindow.configure(background='white')

        self.mainWindow.option_add("*font", "Roboto 9")

        matplotlib.use("TkAgg")

        self.canvas = tk.Canvas(self.mainWindow, width=self.width, height=self.height, bg='white')

        self.btnPlay = tk.Button(self.mainWindow, text="Pause Simulation", command=self.togglePause)
        self.btnPlay.grid(row=0, column=0, sticky="nsew")

        self.btnUpdate = tk.Button(self.mainWindow, text="Pause Render ", command=self.toggleUpdateScreen)
        self.btnUpdate.grid(row=0, column=1, sticky="nsew")

        self.btnStep = tk.Button(self.mainWindow, text="Step Forward", command=self.advanceOneStep, relief=tk.RAISED)
        self.btnStep.grid(row=0, column=2, sticky="nsew")

        self.lastSelectedGraph = tk.StringVar(self.mainWindow)
        self.lastSelectedGraph.set(self.graphOptions[0])  # default value

        self.btnGraphMenu = tk.Menubutton(self.mainWindow, text="Graphs", relief=tk.RAISED)
        self.graphMenu = tk.Menu(self.btnGraphMenu, tearoff=0)

        self.btnGraphMenu.configure(menu=self.graphMenu)

        self.optionNames = self.graphOptions.copy()
        for option in self.graphOptions:
            self.graphMenu.add_checkbutton(label=option, onvalue=option, offvalue=option,
                                           variable=self.lastSelectedGraph,
                                           command=self.updateGraphList, indicatoron=False)
        self.btnGraphMenu.grid(row=0, column=3, sticky="nsew")

        self.btnEnvViewMenu = tk.Menubutton(self.mainWindow, text="Env Color",
                                            relief=tk.RAISED)  # TODO: add more? If so, make like other menus

        self.viewMenu = tk.Menu(self.btnEnvViewMenu, tearoff=0)

        self.btnEnvViewMenu.configure(menu=self.viewMenu)
        self.btnEnvViewMenu.grid(row=0, column=4, sticky="nsew")
        self.envViewMenu = tk.Menu(self.viewMenu, tearoff=0)
        if rules["pollution"]:
            self.envViewMenu.add_checkbutton(label="Color By Pollution", command=self.toggleColorByPollution)

        self.lastSelectedAgentView = tk.StringVar(self.mainWindow)
        self.lastSelectedAgentView.set(self.agentViewOptions[1])  # default value

        self.btnAgentViewMenu = tk.Menubutton(self.mainWindow, text="Agent Color", relief=tk.RAISED)
        self.agentViewMenu = tk.Menu(self.btnAgentViewMenu, tearoff=0)
        self.btnAgentViewMenu.configure(menu=self.agentViewMenu)
        for option in self.agentViewOptions:
            self.agentViewMenu.add_checkbutton(label=option, onvalue=option, offvalue=option,
                                               variable=self.lastSelectedAgentView,
                                               command=self.updateAgentView, indicatoron=True)

        self.viewMenu.add_cascade(label="Agent Color", menu=self.agentViewMenu)
        self.viewMenu.add_cascade(label="Environment Color", menu=self.envViewMenu)

        self.btnStats = tk.Button(self.mainWindow, text="Show Stats", command=self.createStatsWindow)
        self.btnStats.grid(row=0, column=5, sticky="nsew")

        self.canvas.grid(row=1, column=0, columnspan=6, sticky="nsew")

        self.mainWindow.bind("<Escape>", lambda x: self.setQuit())

        self.mainWindow.protocol("WM_DELETE_WINDOW", self.on_mainClosing)

        self.initialDraw()
        self.updateWindow()

    def step(self):
        last_time = time.time()
        # update sugarscape
        if not self.pause:
            self.updateGame()

            # display sugarscape state
            if self.updateScreen:
                self.draw()
            # calculate and display the framerate
            time_now = time.time()
            time_since_last_frame = time_now - last_time
            framerate = int(round(1.0 / time_since_last_frame, 0))

            # display info
            if self.update:
                print("Iteration = ", self.iteration, "; fps = ", framerate, "; Seasons (N,S) = ", self.season,
                      "; Population = ", len(self.agents), " -  press F12 to pause.")

        self.mainWindow.update()

    def updateWindow(self):
        while not self.quit:
            self.step()
        exit(0)


''' 
Main 
'''

if __name__ == '__main__':

    env = Environment(gridSize)

    # add radial food site 
    env.addSugarSite(sites["northeast"], maxCapacity)

    # add radial food site 
    env.addSugarSite(sites["southwest"], maxCapacity)

    if rules["pollution"]:
        env.setPollutionRules(pA, pB, pDiffusionRate, pollutionStartTime, diffusionStartTime)

    if rules["spice"]:
        # add radial food site
        env.addSpiceSite(sites["southeast"], maxCapacity)

        # add radial food site
        env.addSpiceSite(sites["northwest"], maxCapacity)

    env.setHasLimitedLifespan(rules["limitedLifespan"])

    if rules["foresight"]:
        env.setHasForesight(True)
        env.setForesightRange(foresightRange)

    # grow to max capacity
    if rules["grow"]:
        env.grow(maxCapacity)

    if rules["disease"]:
        env.setHasDisease(True)
        env.setImmuneSystemSize(immuneSystemSize)
        env.setDiseaseLength(diseaseLength)
        for i in range(numDiseases):
            env.generateDisease()

    if rules["credit"]:
        env.setLoanRate(loanRate)
        env.setLoanDuration(loanDuration)

    # create a list of agents and place them in env
    agents = []
    for (numAgents, tags, distribution) in distributions:
        for _ in range(numAgents):
            agent = Agent(env)
            if initAgent(agent, tags, distribution):
                env.setAgent(agent.getLocation(), agent)

                if rules["disease"]:
                    agent.createImmuneSystem()
                    for _ in range(numStartingDiseases):
                        agent.addRandomDisease()
                agents.append(agent)

    # Create a view with an env and a list of agents in env
    view = View(screenSize, env, agents)

    # iterate
    view.createWindow()
