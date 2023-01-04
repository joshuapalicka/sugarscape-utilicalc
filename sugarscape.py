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
import matplotlib as mpl

import matplotlib.pyplot as plt

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
    "blue": "#3232FA"
}

# environment
sites = {
    "northeast": (35, 15, 18),
    "southwest": (15, 35, 18),
    "southeast": (40, 40, 14),
    "northwest": (10, 10, 14)
}

maxCapacity = 10  # !!! < or = nbr items in colorSugar array
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
maxAgentVision = 8
initEndowment = 25, 50
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
diseaseLength = 4  # number of bits per disease
numDiseases = 5
numStartingDiseases = 3

# Active settings
agentColorScheme = 4
distributions = [
    (50, tags0, (0, 50, 0, 50)),  # blues
    (50, tags1, (0, 50, 0, 50))]  # reds

rules = {
    "grow": True,
    "seasons": False,
    "moveEat": True,  # move eat and combat need to be exclusive
    "combat": False,
    "limitedLife": False,
    "replacement": False,
    "procreate": True,
    "transmit": False,
    "spice": False,
    "trade": False,
    "credit": False,
    "inheritance": False,
    "disease": False
}

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


def lightenColor(color, amountAtLocation):
    """ lighten color by factor """
    factor = amountAtLocation / maxCapacity
    rgb = hexToRGB(color)
    return RGBToHex(tuple(int(c + (255 - c) * (1 - factor)) for c in rgb))


def initAgent(agent, tags, distribution):
    newLocation = agent.getEnv().getRandomFreeLocation(distribution)
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
        self.wealthWidget, self.metabolismWidget, self.popWidget = None, None, None
        self.window, self.canvas = None, None
        self.height, self.width = screenSize[0] + 50, screenSize[1]
        self.pause = False
        self.updateScreen = True
        self.quit = False
        self.siteSize = screenSize[0] / env.gridWidth
        self.radius = int(self.siteSize * 0.5)
        # init env
        self.env = env
        self.season = ""
        # init agents population
        self.agents = agents
        self.population = []

        self.metabolismMean = []
        self.visionMean = []
        self.tradePriceMean = []
        self.tradeVolumeMean = []
        self.gini = []
        self.numInfectedAgents = []
        self.proportionInfectedAgents = []

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

    def byMetabolism(self, agent):
        if agent.getMetabolism() > 2:
            return colors["red"]
        else:
            return colors["blue"]

    def byVision(self, agent):
        if agent.getVision() > 3:
            return colors["red"]
        else:
            return colors["blue"]

    def byGroup(self, agent):
        #        if bin(agent.getTags()).count('1') > agent.getTagsLength()>>1:
        if agent.getTribe() == 1:
            return colors["red"]
        else:
            return colors["blue"]

    agentColorSchemes = {0: all, 1: bySex, 2: byMetabolism, 3: byVision, 4: byGroup}

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
            if initAgent(agent, tags, self.findDistribution(tags)):
                self.env.setAgent(agent.getLocation(), agent)
            else:
                print("initAgent failed!")
                self.agents.remove(agent)
        else:
            self.agents.remove(agent)

    def hasStarved(self, agent):
        if agent.getSugar() <= 0:
            agent.addLogEntry("Starved due to lack of sugar")
            # free environment
            self.env.setAgent(agent.getLocation(), None)
            # remove or replace agent
            self.removeAgent(agent)
            return True
        if rules["spice"]:
            if agent.getSpice() <= 0:
                agent.addLogEntry("Starved due to lack of spice")
                self.env.setAgent(agent.getLocation(), None)
                self.removeAgent(agent)
                return True
        return False

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
                while True:
                    try:
                        # if a new baby is born, append it to the agents' list
                        self.agents.append(next(mateItr))
                    except StopIteration:
                        break

            if rules["disease"]:
                agent.disease()

            if rules["trade"]:
                trades.extend(agent.trade())

            # TRANSMIT
            if rules["transmit"]:
                agent.transmit()

            agent.setSugar(max(agent.getSugar() - agent.getSugarMetabolism(), 0), "Metabolism")

            if rules["spice"]:
                agent.setSpice(max(agent.getSpice() - agent.getSpiceMetabolism(), 0), "Metabolism")

            if self.hasStarved(agent):
                continue

            # Log agent's parameters
            sugarMetabolism += agent.getSugarMetabolism()
            if rules["spice"]:
                spiceMetabolism += agent.getSpiceMetabolism()
            vision += agent.getVision()

            if not rules["spice"]:
                wealth.append(agent.getSugar())
            else:
                wealth.append(agent.getSugar() + agent.getSpice())

            # DIE
            # increment age
            alive = agent.incAge()
            if rules["limitedLife"] and not alive:
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
        self.tradePriceMean.append(sum(trades) / len(trades)) if len(trades) > 0 else self.tradePriceMean.append(0)
        self.tradeVolumeMean.append(len(trades)) if len(trades) > 0 else self.tradeVolumeMean.append(0)
        self.gini.append(calculateGini(wealth) if numAgents > 0 else 0)

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

        if self.iteration % graphUpdateFrequency == 0:
            if len(self.onGraphs) > 0:
                self.updateGraphs()

    def getFillColor(self, row, col):
        current_agent = env.getAgent((row, col))
        # change color of site depending on what's on it - but only if it wasn't already that color (performance optimization)
        if current_agent:
            fill_color = self.agentColorSchemes[agentColorScheme](self, current_agent)
        else:
            sugarCapacity = env.getSugarCapacity((row, col))
            if not rules["spice"]:
                fill_color = lightenColor(colors["sugar"], sugarCapacity)
            else:
                spiceCapacity = env.getSpiceCapacity((row, col))
                if sugarCapacity >= spiceCapacity:
                    fill_color = lightenColor(colors["sugar"], sugarCapacity)
                elif sugarCapacity < spiceCapacity:
                    fill_color = lightenColor(colors["spice"], spiceCapacity)
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

    def setQuit(self):
        self.quit = True

    def togglePause(self):
        print("Pause: ", self.pause)
        self.pause = not self.pause

    def toggleUpdateScreen(self):
        print("Update: ", self.updateScreen)
        self.updateScreen = not self.updateScreen

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
        self.figs[ax_idx] = (fig, ax)

    def updateWealthPlot(self, ax_idx):
        fig, ax = self.figs[ax_idx]
        ax.clear()
        # create figure
        ax.hist(self.getAgentsWealth(), bins=10)
        ax.set_title("Wealth histogram")
        ax.set_xlabel("Wealth")
        ax.set_ylabel("Number of agents")
        self.figs[ax_idx] = (fig, ax)

    def updateMetabolismVisionPlot(self, ax_idx):
        fig, ax = self.figs[ax_idx]
        ax.clear()
        ax.plot(self.metabolismMean, label="metabolism")
        ax.plot(self.visionMean, label="vision")
        ax.set_title("Agents' metabolism and vision mean values")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mean value")
        ax.legend()
        self.figs[ax_idx] = (fig, ax)

    def updateTradePricePlot(self, ax_idx):
        # create figure
        fig, ax = self.figs[ax_idx]
        ax.clear()
        index = list(range(len(self.tradePriceMean)))
        data, index = boundData(self.tradePriceMean, index)
        ax.scatter(x=index, y=data, label="price", s=4)
        ax.set_title("Mean trade price")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mean price")
        ax.legend()
        self.figs[ax_idx] = (fig, ax)

    def updateTradeVolumePlot(self, ax_idx):
        # create figure
        fig, ax = self.figs[ax_idx]
        ax.clear()
        ax.plot(self.tradeVolumeMean, label="volume")
        ax.set_title("Mean trade volume")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mean volume")
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
        ax.legend()
        self.figs[ax_idx] = (fig, ax)

    def updateInfectedPlot(self, ax_idx):
        fig, ax = self.figs[ax_idx]
        ax.clear()
        ax.plot(range(len(self.proportionInfectedAgents)), self.proportionInfectedAgents)
        ax.set_title("Proportion of Infected Agents to Healthy Agents time series")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Proportion of Infected Agents")
        self.figs[ax_idx] = (fig, ax)

    def populateOptions(self):  # TODO: Add more graphs and options
        # populate graph options depending on the rules
        self.options = []

        self.options.append("Population")
        self.options.append("Wealth")
        self.options.append("Metabolism and vision")
        if rules["trade"]:
            self.options.append("Trade price")
            self.options.append("Trade volume")
            self.options.append("Gini")

        if rules["disease"]:
            self.options.append("Proportion Infected")

        self.offGraphs = self.options.copy()
        self.onGraphs = []

    def updateGraphList(self, *args):
        new_onGraph = self.variable.get()
        if new_onGraph in self.offGraphs:
            self.offGraphs.remove(new_onGraph)
            self.onGraphs.append(new_onGraph)
        elif new_onGraph in self.onGraphs:
            self.onGraphs.remove(new_onGraph)
            self.offGraphs.append(new_onGraph)
        self.updateGraphs()
    def handle_close(self, evt):
        fig = self.figs.pop()
        plt.close(fig[0])
        self.onGraphs.remove(fig[1])
        self.offGraphs.append(fig[1])

        #self.updateGraphList()

    def checkAddFig(self):
        while len(self.onGraphs) > len(self.figs):
            print("Adding fig")
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
            elif self.onGraphs[i] == "Metabolism and vision":
                self.updateMetabolismVisionPlot(i)
            elif self.onGraphs[i] == "Trade price":
                self.updateTradePricePlot(i)
            elif self.onGraphs[i] == "Trade volume":
                self.updateTradeVolumePlot(i)
            elif self.onGraphs[i] == "Gini":
                self.updateGiniPlot(i)
            elif self.onGraphs[i] == "Proportion Infected":
                self.updateInfectedPlot(i)
            self.figs[i][0].canvas.draw()
            self.figs[i][0].canvas.flush_events()

        if totalPlots < len(self.figs):
            for _ in range(len(self.figs) - totalPlots):
                fig = self.figs.pop()
                plt.close(fig[0])

    # the main game loop
    def createWindow(self):  # TODO: if graph window closed, crash occurs
        self.update = True
        plt.ion()
        self.figs = []
        self.populateOptions()
        self.window = tk.Tk()
        self.window.title("Sugarscape")
        self.window.geometry("%dx%d" % (self.width + 5, self.height - 5))
        self.window.resizable(True, True)
        self.window.configure(background='white')

        self.canvas = tk.Canvas(self.window, width=self.width, height=self.height, bg='white')

        self.btnQuit = tk.Button(self.window, text="Quit", command=self.setQuit)
        self.btnQuit.grid(row=0, column=0, sticky="nsew")

        self.btnPlay = tk.Button(self.window, text="Play/Pause", command=self.togglePause)
        self.btnPlay.grid(row=0, column=1, sticky="nsew")

        self.btnUpdate = tk.Button(self.window, text="Update Screen", command=self.toggleUpdateScreen)
        self.btnUpdate.grid(row=0, column=2, sticky="nsew")

        self.variable = tk.StringVar(self.window)
        self.variable.set(self.options[0])  # default value
        self.menubutton = tk.Menubutton(self.window, text="Graphs", relief=tk.RAISED)
        self.menu = tk.Menu(self.menubutton, tearoff=0)
        self.menubutton.configure(menu=self.menu)
        for option in self.options:
            self.menu.add_checkbutton(label=option, onvalue=option, offvalue=option, variable=self.variable,
                                      command=self.updateGraphList)
        self.menubutton.grid(row=0, column=3, sticky="nsew")

        self.canvas.grid(row=1, column=0, columnspan=4, sticky="nsew")

        self.window.bind("<Escape>", lambda x: self.setQuit())

        self.initialDraw()
        self.updateWindow()

    def updateWindow(self):
        while not self.quit:
            last_time = time.time()
            # update sugarscape
            if not self.pause:
                self.updateGame()
                self.iteration += 1
                env.incrementTime()

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
            self.window.update()
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

    if rules["spice"]:
        # add radial food site
        env.addSpiceSite(sites["southeast"], maxCapacity)

        # add radial food site
        env.addSpiceSite(sites["northwest"], maxCapacity)

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
