'''
Created on 2010-04-17

@author: rv

Updated by joshuapalicka
'''

import random
import time
from itertools import product
from environment import Environment
from agent import Agent
from wdgPopulation import WdgPopulation
from wdgWealth import WdgWealth
from wdgAgent import WdgAgent
import tkinter as tk

''' 
initial simulation parameters
'''

# view
screenSize = 600, 600
gridSize = 50, 50
colorBackground = 250, 250, 250

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
maxAgentVision = 5
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

# Active settings
agentColorScheme = 4
distributions = [
    (100, tags0, (0, 50, 0, 50)),  # blues
    (100, tags1, (0, 50, 0, 50))]  # reds
ruleGrow = True
ruleSeasons = False
ruleMoveEat = True  # move eat and combat need to be exclusive
ruleCombat = False
ruleLimitedLife = False
ruleReplacement = False
ruleProcreate = False
ruleTransmit = False
ruleSpice = True
ruleTrade = True
isRandom = False
combatAlpha = 1000000

if not isRandom:
    random.seed(0)

''' 
Global functions
'''


def hexToRGB(hex):
    """ #FFFFFF -> (255, 255, 255) """
    hex = hex.lstrip('#')
    hlen = len(hex)
    return tuple(int(hex[i:i + hlen // 3], 16) for i in range(0, hlen, hlen // 3))


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
    if ruleSpice:
        agent.setSpiceMetabolism(random.randint(1, maxAgentMetabolism))
        agent.setInitialSpiceEndowment(random.randint(initEndowment[0], initEndowment[1]))
    agent.setVision(random.randint(1, maxAgentVision))
    agent.setAge(random.randint(minmaxAgentAge[0], minmaxAgentAge[1]))
    sex = random.randint(0, 1)
    agent.setSex(sex)
    agent.setFertility((random.randint(childbearing[sex][0], childbearing[sex][1]),
                        random.randint(childbearing[sex][2], childbearing[sex][3])))
    if tags == None:
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


''' 
View Class
'''


class View:

    # this gets called first
    def __init__(self, screenSize, env, agents):
        # init view
        self.wealthWidget, self.metabolismWidget, self.popWidget = None, None, None
        self.window, self.canvas = None, None
        self.height, self.width = screenSize[0] + 5, screenSize[1] + 5
        self.update = None
        self.quit = False
        self.siteSize = screenSize[0] / env.gridWidth
        self.radius = int(self.siteSize * 0.5)
        # init env
        self.env = env
        self.season = ""
        # init agents population
        self.agents = agents
        self.population = [len(self.agents)]

        self.metabolismMean = []
        self.visionMean = []
        self.tradePriceMean = []
        self.tradeVolumeMean = []
        self.gini = []

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
        if ruleReplacement:
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
            # free environment
            self.env.setAgent(agent.getLocation(), None)
            # remove or replace agent
            self.removeAgent(agent)
            return True
        if ruleSpice:
            if agent.getSpice() <= 0:
                self.env.setAgent(agent.getLocation(), None)
                self.removeAgent(agent)
                return True
        return False

    # put game update code here
    def updateGame(self):
        # for agents' logs
        sugarMetabolism = 0
        if ruleSpice:
            spiceMetabolism = 0
        vision = 0
        wealth = []

        # execute agents randomly
        random.shuffle(self.agents)

        trades = []
        # run agents' rules
        for agent in self.agents:
            # MOVE
            if ruleMoveEat:
                agent.move()


            # COMBAT
            if ruleCombat:
                killed = agent.combat(combatAlpha)
                # if an agent has been killed, remove it
                if killed:
                    # do not free the environment, someone else is already here
                    self.removeAgent(killed)

            # PROCREATE
            if ruleProcreate and agent.isFertile():
                mateItr = agent.mate()
                while True:
                    try:
                        # if a new baby is born, append it to the agents' list
                        self.agents.append(next(mateItr))
                    except StopIteration:
                        break

            if ruleTrade:
                for trade in agent.trade():
                    trades.append(trade)  # returns a list of trade prices

            # TRANSMIT
            if ruleTransmit:
                agent.transmit()

            agent.setSugar(max(agent.getSugar() - agent.getSugarMetabolism(), 0), "Metabolism")

            if ruleSpice:
                agent.setSpice(max(agent.getSpice() - agent.getSpiceMetabolism(), 0), "Metabolism")

            if self.hasStarved(agent):
                continue

            # Log agent's parameters
            sugarMetabolism += agent.getSugarMetabolism()
            if ruleSpice:
                spiceMetabolism += agent.getSpiceMetabolism()
            vision += agent.getVision()

            if not ruleSpice:
                wealth.append(agent.getSugar())
            else:
                wealth.append(agent.getSugar() + agent.getSpice())

            # DIE
            # increment age
            alive = agent.incAge()
            if ruleLimitedLife and not alive:
                # free environment
                self.env.setAgent(agent.getLocation(), None)
                # remove or replace agent
                self.removeAgent(agent)

        # Log population
        numAgents = len(self.agents)
        self.population.append(numAgents)

        # Calculate and log agents' metabolism and vision mean values
        if numAgents > 0:
            if not ruleSpice:
                self.metabolismMean.append(sugarMetabolism / numAgents)
            else:
                self.metabolismMean.append(((sugarMetabolism + spiceMetabolism) / 2) / float(numAgents))

            self.visionMean.append(vision / float(numAgents))
            self.tradePriceMean.append(sum(trades) / len(trades)) if len(trades) > 0 else self.tradePriceMean.append(0)
            self.tradeVolumeMean.append(len(trades)) if len(trades) > 0 else self.tradeVolumeMean.append(0)
            self.gini.append(calculateGini(wealth))

        # run environment's rules
        if ruleSeasons:
            S = (self.iteration % (2 * seasonPeriod)) / seasonPeriod
            if S < 1:
                # Summer
                self.season = "(summer, winter)"
                if ruleGrow:
                    self.env.growRegion(seasonRegions["north"], growFactor1)
                    self.env.growRegion(seasonRegions["south"], growFactor2)
            else:
                # winter
                self.season = "(winter, summer)"
                if ruleGrow:
                    self.env.growRegion(seasonRegions["north"], growFactor2)
                    self.env.growRegion(seasonRegions["south"], growFactor1)
        elif ruleGrow:
            self.season = "NA"
            self.env.grow(growFactor)

    def draw(self):
        for row, col in product(range(len(self.grid)), range(len(self.grid[0]))):
            agent = env.getAgent((row, col))
            # change color of site depending on what's on it - but only if it wasn't already that color (performance optimization)
            if agent:
                fill_color = self.agentColorSchemes[agentColorScheme](self, agent)
            else:
                sugarCapacity = env.getSugarCapacity((row, col))
                if not ruleSpice:
                    fill_color = lightenColor(colors["sugar"], sugarCapacity)
                else:
                    spiceCapacity = env.getSpiceCapacity((row, col))
                    if sugarCapacity >= spiceCapacity:
                        fill_color = lightenColor(colors["sugar"], sugarCapacity)
                    elif sugarCapacity < spiceCapacity:
                        fill_color = lightenColor(colors["spice"], spiceCapacity)
                    else:
                        fill_color = "white"

            if self.grid[row][col][1] != fill_color:
                self.canvas.itemconfig(self.grid[row][col][0], fill=fill_color)
                self.grid[row][col] = (self.grid[row][col][0], fill_color)

        self.canvas.pack()

    # put drawing code here
    def initialDraw(self):
        # display Sugarscape
        for row, col in product(range(env.gridHeight), range(env.gridWidth)):
            x1 = 5 + (.5 * self.siteSize) + row * self.siteSize - (.5 * self.siteSize)
            y1 = 5 + (.5 * self.siteSize) + col * self.siteSize - (.5 * self.siteSize)
            x2 = 5 + (.5 * self.siteSize) + row * self.siteSize + (.5 * self.siteSize)
            y2 = 5 + (.5 * self.siteSize) + col * self.siteSize + (.5 * self.siteSize)
            agent = env.getAgent((row, col))
            if agent:
                fill_color = self.agentColorSchemes[agentColorScheme](self, agent)

            else:
                # display sugar's capacity
                sugarCapacity = env.getSugarCapacity((row, col))
                if not ruleSpice:
                    fill_color = lightenColor(colors["sugar"], sugarCapacity)
                else:
                    spiceCapacity = env.getSpiceCapacity((row, col))
                    if sugarCapacity >= spiceCapacity:
                        fill_color = lightenColor(colors["sugar"], sugarCapacity)
                    elif sugarCapacity < spiceCapacity:
                        fill_color = lightenColor(colors["spice"], spiceCapacity)
                    else:
                        fill_color = "white"
            self.grid[row][col] = (
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline="#C0C0C0"), fill_color)
        self.canvas.pack()

    def setQuit(self):
        self.quit = True

    def setPause(self):
        print("Pause: ", self.update)
        self.update = not self.update

    def createPopulationPlot(self):
        self.popWidget = WdgPopulation(self.population, "Population time series", 1000, 300)

    def createWealthPlot(self):
        print("Create wealth plot")
        self.wealthWidget = WdgWealth(self.agents, "Wealth histogram", 500, 500)

    def createMetabolismPlot(self):
        print("Create metabolism plot")
        self.metabolismWidget = WdgAgent(self.metabolismMean, self.visionMean,
                                         "Agents' metabolism and vision mean values", 1000, 300)

    # the main game loop
    def createWindow(self):
        self.update = True
        self.window = tk.Tk()
        self.window.title("Sugarscape")
        self.window.geometry("%dx%d" % (self.width + 5, self.height + 5))
        self.window.resizable(True, True)
        self.window.configure(background='white')
        self.canvas = tk.Canvas(self.window, width=self.width, height=self.height, bg='white')

        self.window.bind("<Escape>", lambda x: self.setQuit())
        self.window.bind("<F1>", lambda x: self.createPopulationPlot())
        self.window.bind("<F2>", lambda x: self.createWealthPlot())
        self.window.bind("<F3>", lambda x: self.createMetabolismPlot())
        self.window.bind("<F12>", lambda x: self.setPause())
        self.initialDraw()
        self.updateWindow()

    def updateWindow(self):
        while not self.quit:
            last_time = time.time()
            # update sugarscape
            if self.update:
                self.updateGame()
                self.iteration += 1
                env.incrementTime()

            # display sugarscape state
            self.draw()
            self.window.update()

            if self.popWidget:
                self.popWidget.update(self.population)
                if self.iteration >= 900 and self.iteration % 100 == 0:
                    self.popWidget.makeWider()  # TODO: hacky -- will be fixed (probably by removal of these widgets)

            if self.wealthWidget:
                self.wealthWidget.update(self.agents)

            if self.metabolismWidget:
                self.metabolismWidget.update(self.metabolismMean, self.visionMean)
                if self.iteration >= 900 and self.iteration % 100 == 0:
                    self.metabolismWidget.makeWider()  # TODO: hacky -- will be fixed (probably by removal of these widgets)

            # calculate and display the framerate
            time_now = time.time()
            time_since_last_frame = time_now - last_time
            framerate = int(round(1.0 / time_since_last_frame, 0))

            # display infos
            if self.update:
                print("Iteration = ", self.iteration, "; fps = ", framerate, "; Seasons (N,S) = ", self.season,
                      "; Population = ", len(self.agents), " -  press F12 to pause.")


''' 
Main 
'''

if __name__ == '__main__':

    env = Environment(gridSize)

    # add radial food site 
    env.addSugarSite(sites["northeast"], maxCapacity)

    # add radial food site 
    env.addSugarSite(sites["southwest"], maxCapacity)

    if ruleSpice:
        # add radial food site
        env.addSpiceSite(sites["southeast"], maxCapacity)

        # add radial food site
        env.addSpiceSite(sites["northwest"], maxCapacity)

    # grow to max capacity
    if ruleGrow:
        env.grow(maxCapacity)

    # create a list of agents and place them in env
    agents = []
    for (numAgents, tags, distribution) in distributions:
        for i in range(numAgents):
            agent = Agent(env)
            if initAgent(agent, tags, distribution):
                env.setAgent(agent.getLocation(), agent)
                agents.append(agent)

    # Create a view with an env and a list of agents in env
    view = View(screenSize, env, agents)

    # iterate
    view.createWindow()
