"""
Created on 2010-04-17

@author: rv

Updated/extended by joshuapalicka
"""

import math
import random
import time
from itertools import product
from environment import Environment
from agent import Agent
import tkinter as tk
import json

import matplotlib.pyplot as plt
import matplotlib

''' 
initial simulation parameters
'''
with open('settings.json') as settings_file:
    settings = json.load(settings_file)

# view
screenSize = settings["view"]["screen_size"]["x"], settings["view"]["screen_size"]["y"]
gridSize = settings["view"]["grid_size"]["x"], settings["view"]["grid_size"]["y"]
colorBackground = settings["view"]["background_colors"]["R"], settings["view"]["background_colors"]["G"], \
                  settings["view"]["background_colors"]["B"]
graphUpdateFrequency = settings["view"]["graph_update_frequency"]  # graphs update every graphUpdateFrequency frames

# display colors - can be changed here and will update in the GUI
colors = {
    "sugar": settings["view"]["env_color_hashes"]["sugar"],
    "spice": settings["view"]["env_color_hashes"]["spice"],
    "both": settings["view"]["env_color_hashes"]["both"],
    "red": settings["view"]["env_color_hashes"]["red"],
    "pink": settings["view"]["env_color_hashes"]["pink"],
    "blue": settings["view"]["env_color_hashes"]["blue"],
    "pollution": settings["view"]["env_color_hashes"]["pollution"]
}

# environment
sites = {
    "northeast": (
        settings["env"]["sites"]["ne"]["x"], settings["env"]["sites"]["ne"]["y"],
        settings["env"]["sites"]["ne"]["radius"]),
    "southwest": (
        settings["env"]["sites"]["sw"]["x"], settings["env"]["sites"]["sw"]["y"],
        settings["env"]["sites"]["sw"]["radius"]),
    "southeast": (
        settings["env"]["sites"]["se"]["x"], settings["env"]["sites"]["se"]["y"],
        settings["env"]["sites"]["se"]["radius"]),
    "northwest": (
        settings["env"]["sites"]["nw"]["x"], settings["env"]["sites"]["nw"]["y"],
        settings["env"]["sites"]["nw"]["radius"])
}

maxCapacity = settings["env"]["max_capacity"]

rule_params = settings["rule_params"]

seasonPeriod = rule_params["seasons"]["period"]
seasonRegionsDict = rule_params["seasons"]["regions"]
seasonRegions = {
    "north": (
        seasonRegionsDict["north"]["start_x"], seasonRegionsDict["north"]["start_y"],
        seasonRegionsDict["north"]["end_x"],
        seasonRegionsDict["north"]["end_y"]),
    "south": (
        seasonRegionsDict["south"]["start_x"], seasonRegionsDict["south"]["start_y"],
        seasonRegionsDict["south"]["end_x"],
        seasonRegionsDict["south"]["end_y"]),
}

growFactor = settings["env"]["growth_factor"]["grow_factor"]
growFactorOnSeason = settings["env"]["growth_factor"]["grow_factor_on_season"]
growFactorOffSeasonDivisor = settings["env"]["growth_factor"]["grow_factor_off_season_divisor"]
growFactorOffSeason = float(growFactorOnSeason) / growFactorOffSeasonDivisor

# agents
# agentColorScheme: Agents colour meaning = 0:all, 1:bySex, 2:byMetabolism, 3:byVision, 4:byGroup
maxAgentMetabolism = settings["agents"]["max_agent_metabolism"]
maxAgentVision = settings["agents"]["max_agent_vision"]
initEndowment = settings["agents"]["initial_endowments"]["sugar"], settings["agents"]["initial_endowments"]["spice"]
minmaxAgentAge = settings["agents"]["death_age_range"]["min"], settings["agents"]["death_age_range"]["max"]

female = 0
male = 1
fert_ages = settings["agents"]["fertility_age_ranges"]
fertility = [(fert_ages["male"]["min_start"], fert_ages["male"]["max_start"], fert_ages["male"]["min_end"],
              fert_ages["male"]["max_end"]),
             (fert_ages["female"]["min_start"], fert_ages["female"]["max_start"], fert_ages["female"]["min_end"],
              fert_ages["female"]["max_end"])]
childbearing = fertility[0], fertility[1]  # female , male

tagsLength = rule_params["tags"]["tags_length"]  # must be odd
tags0 = rule_params["tags"]["tags0"]
tags1 = 2 ** tagsLength - 1

loanRate = rule_params["loans"]["rate"]
loanDuration = rule_params["loans"]["duration"]

# There exists 2^(diseaseLength) unique diseases and 2^immuneSystemSize unique immune systems
immuneSystemSize = rule_params["disease"]["immune_system_size"]  # number of bits per immune system
diseaseLength = rule_params["disease"]["disease_length"]  # number of bits per disease
numDiseases = rule_params["disease"]["num_diseases"]
numStartingDiseases = rule_params["disease"]["num_starting_diseases"]

selfInterestScale = rule_params["utilicalc"]["self_interest_scale"]

rules = settings["rules"]

if rules["tags"]:
    agentColorScheme = 4

else:
    agentColorScheme = 1

if rules["pollution"]:
    pA = rule_params["pollution"]["A"]  # production pollution
    pB = rule_params["pollution"]["B"]  # consumption pollution
    pDiffusionRate = rule_params["pollution"]["diffusion_rate"]
    pollutionStartTime = rule_params["pollution"]["pollution_start_time"]
    diffusionStartTime = rule_params["pollution"]["diffusion_start_time"]

if rules["foresight"]:
    foresightRange = (rule_params["foresight"]["range_start"], rule_params["foresight"]["range_end"])

totalAgents = settings["env"]["total_agents"]["blue"], settings["env"]["total_agents"][
    "red"]  # number of blue agents, number of red agents

agent_dist = settings["env"]["agent_distributions"]
distributions = [
    (totalAgents[0], tags0, (agent_dist["blue"]["x_min"], agent_dist["blue"]["x_max"], agent_dist["blue"]["y_min"],
                             agent_dist["blue"]["y_max"])),  # blues
    (totalAgents[1], tags1, (agent_dist["red"]["x_min"], agent_dist["red"]["x_max"], agent_dist["red"]["y_min"],
                             agent_dist["red"]["y_max"]))]  # reds

boundGraphData = settings["view"][
    "bound_graph_data"]  # if False, graphs will not be scaled to numDeviations std devs from the mean
numDeviations = settings["view"][
    "bound_to_n_deviations"]  # set graph bounds to be within numDeviations standard deviations of the mean if boundGraphs is True

isRandom = settings["env"]["is_random"]

if rules["combat"]:
    combatAlpha = rule_params["combat"]["alpha"]

if not isRandom:
    random.seed(settings["env"]["random_seed"])


class ConflictingRuleException(Exception):
    def __init__(self, rule1, rule2):
        self.message = str(rule1) + " and " + str(rule2) + " cannot be used together"
        super().__init__(self.message)


class MissingRuleException(Exception):
    def __init__(self, missingRule, enabledRule):
        self.message = str(enabledRule) + " is enabled, but needs " + str(missingRule) + " to function properly"
        super().__init__(self.message)


''' 
Global functions
'''


def ruleCheck():
    if rules["moveEat"]:
        if rules["combat"]:
            raise ConflictingRuleException("moveEat", "combat")
        if rules["utilicalc"]:
            raise ConflictingRuleException("moveEat", "utilicalc")

    if rules["combat"]:
        if not rules["tags"]:
            raise MissingRuleException("tags", "combat")
        if rules["spice"]:
            raise ConflictingRuleException("combat", "spice")

    if rules["pollution"]:
        if rules["spice"]:
            raise ConflictingRuleException("pollution", "spice")

    if not rules["spice"]:
        if rules["trade"]:
            raise MissingRuleException("spice", "trade")
        if rules["foresight"]:
            raise MissingRuleException("spice", "foresight")
        if rules["credit"]:
            raise MissingRuleException("spice", "credit")
        if rules["inheritance"]:
            raise MissingRuleException("spice", "inheritance")


# Changes a hex color to RGB, as tkinter uses hex colors, but it's easier for me to work with RGB
def hexToRGB(colorHex):
    """ #FFFFFF -> (255, 255, 255) """
    colorHex = colorHex.lstrip('#')
    colorHexLen = len(colorHex)
    return tuple(int(colorHex[i:i + colorHexLen // 3], 16) for i in range(0, colorHexLen, colorHexLen // 3))


# Changes RGB back to hex
def RGBToHex(rgb):
    """ (255, 255, 255) -> #FFFFFF """
    return '#%02x%02x%02x' % rgb


# Lighten a hex color by a factor. The factor is the ratio of the max amount of sugar or spice the
# simulation allows, to the current number at the site
def lightenColorByCapacity(color, amountAtLocation):
    """ lighten color by factor """
    factor = amountAtLocation / maxCapacity
    rgb = hexToRGB(color)
    return RGBToHex(tuple(int(c + (255 - c) * (1 - factor)) for c in rgb))


# Lightens a hex color by a factor: x / maxX (safe to use with x=0)
def lightenColorByX(color, x, maxX):
    if maxX == 0:
        raise ValueError("maxX cannot be 0")
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


# Calculates the gini coefficient of a list of agent wealth values. Used to measure wealth inequality
def calculateGini(wealth):
    """ Calculate the Gini coefficient of a list of wealth values """
    # based on https://planspace.org/2013/06/21/how-to-calculate-gini-coefficient-from-raw-data-in-python/
    wealth = sorted(wealth)
    height, area = 0, 0
    for value in wealth:
        height += value
        area += height - value / 2
    fair_area = height * len(wealth) / 2
    if fair_area == 0:
        return 1
    return (fair_area - area) / fair_area


def getStandardDeviation(data):
    mean = sum(data) / len(data)
    variance = sum([(x - mean) ** 2 for x in data]) / len(data)
    return math.sqrt(variance)


# Uses boundGraphData and numDeviations variables to set the graph bounds
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


def hasStarved(agent):
    if agent.getSugar() <= 0.1:
        agent.addLogEntry("Starved due to lack of sugar")
        # free environment
        agent.setAlive(False)
    if rules["spice"]:
        if agent.getSpice() <= 0.1:
            agent.addLogEntry("Starved due to lack of spice")
            agent.setAlive(False)


# Following functions are used to update the view. One of these functions (at a time) determines the color of each agent
# The function that is used is determined by which Agent Color is selected in the GUI
def all(agent):
    return colors["red"]


def bySex(agent):
    if agent.getSex() == female:
        return colors["pink"]
    else:
        return colors["blue"]


def bySugarMetabolism(agent):
    return lightenColorByX(colors["red"], agent.getSugarMetabolism(), maxAgentMetabolism)


def bySpiceMetabolism(agent):
    return lightenColorByX(colors["red"], agent.getSpiceMetabolism(), maxAgentMetabolism)


def byVision(agent):
    return lightenColorByX(colors["red"], agent.getVision(), maxAgentVision)


def byGroup(agent):
    #        if bin(agent.getTags()).count('1') > agent.getTagsLength()>>1:
    if agent.getTribe() == 1:
        return colors["red"]
    else:
        return colors["blue"]


def byAge(agent):
    return lightenColorByX(colors["red"], agent.getAge(), minmaxAgentAge[1])


def byWealth(agent):
    return lightenColorByX(colors["red"], agent.getWealth(), 100)


def byNumberOfDiseases(agent):
    return lightenColorByX(colors["red"], agent.getNumAfflictedDiseases(), numDiseases)


# determine how to distribute agents on the screen
def findDistribution(tags):
    getTribe = lambda x, y: round(float(bin(x).count('1')) / float(y))
    tribe = getTribe(tags, tagsLength)
    for (n, t, d) in distributions:
        if t is not None and getTribe(t, tagsLength) == tribe:
            # found a distribution for tags
            return d
    else:
        # or return best guess
        return d


''' 
View Class
'''


class View:

    # this gets called first
    def __init__(self, screenSize, env, agents):

        # init view

        self.buttonFrame = None
        self.followAgentId = None
        self.followAgent = False
        self.locationStatsTextBox = None
        self.btnSelectAgent = None

        self.btnPrintLog = None
        self.selectedRow = None
        self.selectedColumn = None
        self.locationStats = None
        self.locationStatsWindow = None
        self.update = None
        self.stats = {}
        self.wealthWidget, self.metabolismWidget, self.popWidget = None, None, None
        self.mainWindow, self.canvas, self.locationStatsCanvas = None, None, None
        self.height, self.width = screenSize[0] + 40, screenSize[1] + 5
        self.pause = settings["view"]["start_paused"]
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

        # init graph data vars
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
        self.locationStatsGrid = [[(None, None) for __ in range(3)] for __ in range(3)]

        # init tkinter GUI items and variables
        self.figs = None
        self.btnStats = None
        self.agentViewMenu = None
        self.btnAgentViewMenu = None
        self.lastSelectedAgentView = None
        self.envViewMenu = None
        self.viewMenu = None
        self.btnEnvViewMenu = None
        self.optionNames = None
        self.graphMenu = None
        self.btnGraphMenu = None
        self.lastSelectedGraph = None
        self.btnStep = None
        self.btnUpdate = None
        self.btnPlay = None
        self.statsTextBox = None
        self.agentViewOptions = None

    # Maps the agent color scheme number to which function determines the color of each agent
    agentColorSchemes = {0: all, 1: bySex, 2: bySugarMetabolism, 3: byVision, 4: byGroup, 5: byAge, 6: byWealth,
                         7: bySpiceMetabolism, 8: byNumberOfDiseases}

    # replace or remove agent, and determine if it's necessary to split wealth to children
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
            if initAgent(newAgent, tags, findDistribution(tags)):
                self.env.setAgent(newAgent.getLocation(), newAgent)
                self.agents.append(newAgent)

        if agent in self.agents:
            self.agents.remove(agent)

    # Following function runs once per iteration, and is the main driver of action on the Sugarscape
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

            if rules["utilicalc"]:
                deadAgent = agent.utilicalcMove()
                if deadAgent:
                    self.removeAgent(deadAgent)

            # COMBAT
            if rules["combat"] and not rules["utilicalc"]:
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
                hasStarved(agent)

            if agent.isAlive():
                # increment age
                if rules["limitedLifespan"]:
                    agent.setAlive(agent.incAge())

            else:
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
            self.percentBlueTags.append((numBlue / float(numAgents) * 100) if numAgents > 0 else 0)

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
                    self.env.growRegion(seasonRegions["north"], growFactorOnSeason)
                    self.env.growRegion(seasonRegions["south"], growFactorOffSeason)
            else:
                # winter
                self.season = "(winter, summer)"
                if rules["grow"]:
                    self.env.growRegion(seasonRegions["north"], growFactorOffSeason)
                    self.env.growRegion(seasonRegions["south"], growFactorOnSeason)

        elif rules["grow"]:
            self.season = "NA"
            self.env.grow(growFactor)

        if rules["pollution"]:
            if self.iteration >= diffusionStartTime and self.iteration % pDiffusionRate == 0:
                self.env.spreadPollution()

        # updates graphs if there are any and the iteration is a multiple of the update rate
        if self.iteration % graphUpdateFrequency == 0:
            if len(self.onGraphs) > 0:
                self.updateGraphs()

        self.iteration += 1
        env.incrementTime()
        self.updateStatsWindow()
        self.updateLocationStatsWindow()
        self.drawLocationCanvas()

    # Determines which color each square should be
    def getFillColor(self, row, col):
        if self.env.isLocationValid((row, col)):
            current_agent = env.getAgent((row, col))

            # change color of site depending on what's on it
            if current_agent:
                fillColor = self.agentColorSchemes[self.agentColorScheme](current_agent)
            elif self.colorByPollution:
                fillColor = lightenColorByX(colors["pollution"], env.getPollutionAtLocation((row, col)), 20)
            else:
                sugarCapacity = env.getSugarAmt((row, col))
                if not rules["spice"]:
                    fillColor = lightenColorByCapacity(colors["sugar"], sugarCapacity)
                else:
                    spiceCapacity = env.getSpiceAmt((row, col))
                    if sugarCapacity >= spiceCapacity:
                        fillColor = lightenColorByCapacity(colors["sugar"], sugarCapacity)
                    elif sugarCapacity < spiceCapacity:
                        fillColor = lightenColorByCapacity(colors["spice"], spiceCapacity)
                    else:
                        fillColor = "white"
        else:
            fillColor = "grey"
        return fillColor

    def draw(self):
        for row, col in product(range(len(self.grid)), range(len(self.grid[0]))):
            fillColor = self.getFillColor(row, col)
            if self.grid[row][col][
                1] != fillColor:  # only change fill color if the site wasn't already that color (performance optimization)
                self.canvas.itemconfig(self.grid[row][col][0], fill=fillColor)
                self.grid[row][col] = (self.grid[row][col][0], fillColor)

    # creates tkinter rectangle object for each square in the environment
    def initialDraw(self):
        # display Sugarscape
        for row, col in product(range(env.gridHeight), range(env.gridWidth)):
            x1 = 5 + (.5 * self.siteSize) + row * self.siteSize - (.5 * self.siteSize)
            y1 = 5 + (.5 * self.siteSize) + col * self.siteSize - (.5 * self.siteSize)
            x2 = 5 + (.5 * self.siteSize) + row * self.siteSize + (.5 * self.siteSize)
            y2 = 5 + (.5 * self.siteSize) + col * self.siteSize + (.5 * self.siteSize)

            fillColor = self.getFillColor(row, col)
            self.grid[row][col] = (
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fillColor, outline="#C0C0C0"), fillColor
            )

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

    def on_locationStatsClosing(self):
        self.locationStatsWindow.destroy()
        self.locationStatsWindow = None

    # Make the stats dictionary for the stats window
    def makeStatsDict(self):
        round_to = 2
        self.stats = {
            "Iteration": self.iteration,
            "Average Population": round((sum(self.population) / len(self.population)) if len(self.agents) > 0 else 0,
                                        round_to),
            "Metabolism Mean": round(sum(self.metabolismMean) / len(self.metabolismMean), round_to),
            "Vision Mean": round(sum(self.visionMean) / len(self.visionMean), round_to),
            "Gini Coefficient": round(self.gini[-1], round_to),
            "Average Gini": round(sum(self.gini) / len(self.gini), round_to),
            "Average Wealth": round((sum(self.getAgentsWealth()) / len(self.agents)) if len(self.agents) > 0 else 0,
                                    round_to)
        }

        if rules["trade"]:
            self.stats["Trade Price Mean"] = round(sum(self.tradePriceMean) / len([filter(None, self.tradePriceMean)]),
                                                   round_to)  # Adding filter(None, ...) to remove 0s from the list
            self.stats["Trade Volume Mean"] = round(sum(self.tradeVolumeMean) / self.iteration, round_to)
        if rules["foresight"]:
            self.stats["Foresight Mean"] = round(sum(self.foresightMean) / len(self.foresightMean), round_to)
        if rules["tags"]:
            self.stats["Percent Blue Tags"] = round(sum(self.percentBlueTags) / len(self.percentBlueTags), round_to)
        if rules["disease"]:
            self.stats["Number of Infected Agents"] = round(sum(self.numInfectedAgents) / len(self.numInfectedAgents),
                                                            round_to)
            self.stats["Proportion of Infected Agents"] = round(
                sum(self.proportionInfectedAgents) / len(self.proportionInfectedAgents), round_to)

    def makeLocationStatsDict(self):
        if self.followAgent:
            selectedAgent = self.env.findAgentById(self.followAgentId)
            if selectedAgent:
                self.selectedColumn, self.selectedRow = self.env.findAgentById(self.followAgentId).getLocation()
            else:
                self.selectAgent()
                return

        row = self.selectedColumn
        col = self.selectedRow
        self.locationStats = {"agent": {}, "location": {
            "x": row,
            "y": col,
            "sugar": 0,
            "spice": 0
        }}

        self.locationStats["location"]["sugar"] = self.env.getSugarAmt((col, row))
        if rules["spice"]:
            self.locationStats["location"]["spice"] = self.env.getSpiceAmt((col, row))
        if rules["pollution"]:
            self.locationStats["location"]["pollution"] = self.env.getPollutionAtLocation((row, col))

        agentAtClick = self.env.getAgent((row, col))
        agent = agentAtClick
        if agentAtClick:
            self.locationStats["agent"]["obj"] = agent

        self.locationStats["agent"]["id"] = agent.getId() if agent else 0
        if rules["limitedLifespan"]:
            self.locationStats["agent"]["age"] = agent.getAge() if agent else 0
            self.locationStats["agent"]["maxAge"] = agent.getMaxAge() if agent else 0
        self.locationStats["agent"]["sex"] = agent.getSex() if agent else 0
        self.locationStats["agent"]["vision"] = agent.getVision() if agent else 0
        self.locationStats["agent"]["sugar"] = agent.getSugar() if agent else 0
        self.locationStats["agent"]["sugarMetabolism"] = agent.getSugarMetabolism() if agent else 0
        if rules["spice"]:
            self.locationStats["agent"]["spice"] = agent.getSpice() if agent else 0
            self.locationStats["agent"]["spiceMetabolism"] = agent.getSpiceMetabolism() if agent else 0
        if rules["procreate"]:
            self.locationStats["agent"]["children"] = len(agent.getChildren()) if agent else 0
        if rules["tags"]:
            self.locationStats["agent"]["tags"] = agent.getTags() if agent else 0
        if rules["disease"]:
            agent.getNumAfflictedDiseases() if agent else 0

    def drawLocationCanvas(self):
        lCSiteSize = self.siteSize * 2
        if self.locationStatsCanvas:
            for row, col in product(range(3), range(3)):
                x1 = 5 + (.5 * lCSiteSize) + row * lCSiteSize + (.5 * lCSiteSize)
                y1 = 5 + (.5 * lCSiteSize) + col * lCSiteSize + (.5 * lCSiteSize)
                x2 = 5 + (.5 * lCSiteSize) + row * lCSiteSize - (.5 * lCSiteSize)
                y2 = 5 + (.5 * lCSiteSize) + col * lCSiteSize - (.5 * lCSiteSize)

                fillColor = self.getFillColor(self.selectedColumn + (row - 1), self.selectedRow + (col - 1))

                centerBox = row == 1 and col == 1

                outline = "#C0C0C0"
                if centerBox and fillColor != "grey":
                    outline = "#000000"
                elif fillColor == "grey":
                    outline = ""
                self.locationStatsGrid[row][col] = (
                    self.locationStatsCanvas.create_rectangle(x1, y1, x2, y2,
                                                              fill=fillColor,
                                                              outline=outline,
                                                              width=2 if centerBox else 1,
                                                              tags="centerSquare" if centerBox else "otherSquare"
                                                              )
                )
                self.locationStatsCanvas.tag_raise("centerSquare")

    def onClick(self, event):
        squareAtX = event.x - 5  # subtract 5 to account for small amount of padding on grid
        squareAtY = event.y - 5
        squareSize = screenSize[0] // gridSize[0]
        y = squareAtY // squareSize
        x = squareAtX // squareSize

        self.selectedColumn = x
        self.selectedRow = y

        self.makeLocationStatsDict()
        self.createLocationStatsWindow()
        self.drawLocationCanvas()

    def updateStatsWindow(self):
        if self.statsWindow:
            self.makeStatsDict()
            self.statsTextBox.configure(state='normal')
            self.statsTextBox.delete("1.0", tk.END)
            for key, value in self.stats.items():
                self.statsTextBox.insert(tk.END, key + ": " + str(value) + "\n")
            self.statsTextBox.configure(state='disabled')
            self.statsTextBox.pack()

    def createStatsWindow(self):
        if not self.statsWindow:
            self.statsWindow = tk.Tk()
            self.makeStatsDict()
            self.statsWindow.option_add("*font", "Roboto 14")
            self.statsTextBox = tk.Text(self.statsWindow, state='disabled', height=(1.2 * len(self.stats.keys())),
                                        width=25)
            self.statsWindow.title("Stats")
            self.statsWindow.protocol("WM_DELETE_WINDOW", self.on_statsClosing)
            self.updateStatsWindow()

        else:
            self.statsWindow.destroy()
            self.statsWindow = None

    def printSelectedAgentLog(self):
        if "obj" in self.locationStats["agent"]:
            self.locationStats["agent"]["obj"].printLog()
        else:
            print("No Currently Selected Agent")

    def updateLocationStatsWindow(self):
        if self.locationStatsWindow:
            self.makeLocationStatsDict()

            self.locationStatsTextBox = tk.Text(self.locationStatsWindow, state='disabled', width=20)

            self.locationStatsTextBox.configure(state='normal')

            self.locationStatsTextBox.delete("1.0", tk.END)

            self.locationStatsTextBox.insert(tk.END, "Location:" + "\n")
            for key, value in self.locationStats["location"].items():
                self.locationStatsTextBox.insert(tk.END, key + ": " + str(value) + "\n")

            self.locationStatsTextBox.insert(tk.END, "\n")
            self.locationStatsTextBox.insert(tk.END, "Agent:" + "\n")

            for key, value in self.locationStats["agent"].items():
                if key != "obj":
                    self.locationStatsTextBox.insert(tk.END, key + ": " + str(value) + "\n")

            self.locationStatsTextBox.configure(state='disabled')

            self.btnPrintLog.pack(side="left", fill=tk.BOTH, expand=True)
            self.btnSelectAgent.pack(side="left", fill=tk.BOTH, expand=True)

            self.locationStatsTextBox.grid(row=1, column=0, columnspan=2, sticky="n")
            self.locationStatsCanvas.grid(row=2, column=0, columnspan=2, sticky="n")
            self.buttonFrame.grid(row=3, column=0, columnspan=2)

    def selectAgent(self):
        if "obj" in self.locationStats["agent"]:
            self.followAgent = not self.followAgent
            if self.followAgent:
                self.followAgentId = self.locationStats["agent"]["obj"].getId()
            else:
                self.makeLocationStatsDict()
                self.followAgentId = None
            self.btnSelectAgent.config(text="Stop Select" if self.followAgent else "Select Agent")
        else:
            print("No agent exists at selection")

    def createLocationStatsWindow(self):
        if not self.locationStatsWindow:
            self.locationStatsWindow = tk.Tk()
            self.buttonFrame = tk.Frame(self.locationStatsWindow)


            self.locationStatsWindow.option_add("*font", "Roboto 14")
            self.locationStatsCanvas = tk.Canvas(self.locationStatsWindow, bg='white')

            self.btnPrintLog = tk.Button(self.buttonFrame, text="Print Agent Log",
                                         command=self.printSelectedAgentLog)

            self.btnSelectAgent = tk.Button(self.buttonFrame, text="Select Agent",
                                            command=self.selectAgent)

            self.locationStatsWindow.title("Location Stats")
            self.locationStatsWindow.protocol("WM_DELETE_WINDOW", self.on_locationStatsClosing)
        self.updateLocationStatsWindow()

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

        self.btnPlay = tk.Button(self.mainWindow, text="Play Simulation", command=self.togglePause)
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

        self.canvas.bind("<Button-1>", self.onClick)
        self.initialDraw()
        self.updateWindow()

    def step(self):
        last_time = time.time()
        # update Sugarscape
        if not self.pause:
            self.updateGame()

            # display Sugarscape state
            if self.updateScreen:
                self.draw()
            # calculate and display the framerate
            time_now = time.time()
            time_since_last_frame = time_now - last_time
            framerate = int(round(1.0 / time_since_last_frame, 0))

            # display info
            if self.update:
                if rules["seasons"]:
                    print("Iteration = ", self.iteration, "| fps = ", framerate, "| Seasons (N,S) = ", self.season,
                          "| Population = ", len(self.agents))
                else:
                    print("Iteration = ", self.iteration, "| fps = ", framerate, "| Population = ", len(self.agents))

        self.mainWindow.update()

    def updateWindow(self):
        while not self.quit:
            self.step()
        exit(0)


''' 
Main 
'''

if __name__ == '__main__':

    ruleCheck()

    env = Environment(gridSize)

    # add radial food site 
    env.addSugarSite(sites["northeast"], maxCapacity)

    # add radial food site 
    env.addSugarSite(sites["southwest"], maxCapacity)

    if rules["combat"]:
        env.setHasCombat(True)
        env.setCombatAlpha(combatAlpha)

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
        env.setHasCredit(True)
        env.setLoanRate(loanRate)
        env.setLoanDuration(loanDuration)

    # create a list of agents and place them in env
    agentList = []
    for (nAgents, tags, distribution) in distributions:
        for _ in range(nAgents):
            newAgent = Agent(env)
            if initAgent(newAgent, tags, distribution):
                env.setAgent(newAgent.getLocation(), newAgent)

                if rules["disease"]:
                    newAgent.createImmuneSystem()
                    for _ in range(numStartingDiseases):
                        newAgent.addRandomDisease()
                agentList.append(newAgent)

    if rules["utilicalc"]:
        env.setSelfInterestScale(selfInterestScale)

    # Create a view with an env and a list of agents in env
    view = View(screenSize, env, agentList)

    # iterate
    view.createWindow()
