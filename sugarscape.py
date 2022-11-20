'''
Created on 2010-04-17

@author: rv
'''

import random
import time
from itertools import product
from environment import Environment
from agent import Agent
from wdgPopulation import WdgPopulation
from wdgWealth import WdgWealth
from wdgAgent import WdgAgent
from tkinter import *

''' 
initial simulation parameters
'''

# view
screenSize = 600, 600
colorBackground = 250, 250, 250

colorSugar = (("#FAFAB4",
               "#FAFAC8",
               "#FAFAB4",
               "#FAFAA0",
               "#FAFA8C",
               "#FAFA78",
               "#FAFA64",
               "#FAFA50",
               "#FAFA3C",
               "#FAFA28"))

colorRed = "#FA3232"

colorPink = "#FA32FA"

colorBlue = "#3232FA"

# environment
gridSize = 50, 50 # TODO: test changes in grid size with perf.
northSite = 40, 10, 20
southSite = 15, 35, 20 # TODO: balance?
maxCapacity = 10  # !!! < or = nbr items in colorSugar array
seasonPeriod = 50
northRegion = 0, 0, 49, 24
southRegion = 0, 25, 49, 49
growFactor = 1
growFactor1 = 1
growFactor2 = float(growFactor1) / 8

# agents
# agentColorScheme: Agents colour meaning = 0:all, 1:bySexe, 2:byMetabolism, 3:byVision, 4:byGroup
maxAgentMetabolism = 4
maxAgentVision = 6
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

''' settings for Evolution from random distribution
agentColorScheme = 0        
distributions = [(400, None, (0, 50, 0, 50))] 
ruleGrow = True
ruleSeasons = False
ruleMoveEat = True
ruleCombat = False
ruleLimitedLife = False
ruleReplacement = False
ruleProcreate = False
ruleTransmit = False'''

''' settings for Emergent waves migration
agentColorScheme = 0        
distributions = [(300, None, (0, 20, 30, 50))]
ruleGrow = True
ruleSeasons = False
ruleMoveEat = True
ruleCombat = False
ruleLimitedLife = False
ruleReplacement = False
ruleProcreate = False
ruleTransmit = False'''

''' settings for Seasonal migration
agentColorScheme = 0        
distributions = [(400, None, (0, 50, 0, 50))] 
ruleGrow = True
ruleSeasons = True
ruleMoveEat = True
ruleCombat = False
ruleLimitedLife = False
ruleReplacement = False
ruleProcreate = False
ruleTransmit = False'''

''' settings for societal evolution
agentColorScheme = 3       
distributions = [(300, None, (0, 50, 0, 50))] 
ruleGrow = True
ruleSeasons = False
ruleMoveEat = True
ruleCombat = False
ruleLimitedLife = True
ruleReplacement = False
ruleProcreate = True
ruleTransmit = False'''

''' settings for Cultural transmission
agentColorScheme = 4
distributions = [(200, tags0, (0, 50, 0, 50)), (200, tags1, (0, 50, 0, 50))]
ruleGrow = True
ruleSeasons = False
ruleMoveEat = True
ruleCombat = False
ruleLimitedLife = False
ruleReplacement = False
ruleProcreate = False
ruleTransmit = True'''

'''# settings for Combat with alpha = infinite
agentColorScheme = 4
distributions = [
    (300, tags0, (0, 20, 30, 50)),  # blues
    (300, tags1, (30, 50, 0, 20))]  # reds
ruleGrow = True
ruleSeasons = False
ruleMoveEat = False
ruleCombat = True
ruleLimitedLife = False
ruleReplacement = False
ruleProcreate = False
ruleTransmit = False
combatAlpha = 1000000'''

'''# settings for Trench war with alpha = 2
agentColorScheme = 4
distributions = [
    (300, tags0, (0, 20, 30, 50)),  # blues
    (300, tags1, (30, 50, 0, 20))]  # reds
ruleGrow = True
ruleSeasons = False
ruleMoveEat = False
ruleCombat = True
ruleLimitedLife = False
ruleReplacement = True
ruleProcreate = False
ruleTransmit = False
combatAlpha = 2'''

'''# settings for Combat eliminates Waves with alpha = 2
agentColorScheme = 4
distributions = [
    (300, tags0, (0, 20, 30, 50)),  # blues
    (300, tags1, (30, 50, 0, 20))]  # reds
ruleGrow = True
ruleSeasons = False
ruleMoveEat = False
ruleCombat = True
ruleLimitedLife = False
ruleReplacement = False
ruleProcreate = False
ruleTransmit = False
combatAlpha = 2'''

'''# settings for Combat and cultural transmission
agentColorScheme = 4
distributions = [
    (300, tags0, (0, 20, 30, 50)),  # blues
    (300, tags1, (30, 50, 0, 20))]  # reds
ruleGrow = True
ruleSeasons = False
ruleMoveEat = False
ruleCombat = True
ruleLimitedLife = False
ruleReplacement = False
ruleProcreate = False
ruleTransmit = True
combatAlpha = 1000000'''

# settings for Proto-History
agentColorScheme = 4
distributions = [
    (25, tags0, (0, 50, 0, 50)),  # blues
    (25, tags1, (0, 50, 0, 50))]  # reds
ruleGrow = True
ruleSeasons = False
ruleMoveEat = True
ruleCombat = False
ruleLimitedLife = True
ruleReplacement = False
ruleProcreate = True
ruleTransmit = True
combatAlpha = 1000000

fps = 10

''' 
Global functions
'''


def initAgent(agent, tags, distribution):
    newLocation = agent.getEnv().getRandomFreeLocation(distribution)
    if newLocation == None:
        return False
    agent.setLocation(newLocation)
    agent.setMetabolism(random.randint(1, maxAgentMetabolism))
    agent.setVision(random.randint(1, maxAgentVision))
    agent.setInitialEndowment(random.randint(initEndowment[0], initEndowment[1]))
    agent.setAge(random.randint(minmaxAgentAge[0], minmaxAgentAge[1]))
    sexe = random.randint(0, 1)
    agent.setSexe(sexe)
    agent.setFertility((random.randint(childbearing[sexe][0], childbearing[sexe][1]),
                        random.randint(childbearing[sexe][2], childbearing[sexe][3])))
    if tags == None:
        tags = random.getrandbits(tagsLength)
    agent.setTags((tags, tagsLength))
    return True


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
        # init time
        self.iteration = 0
        self.grid = [[0 for __ in range(env.gridWidth)] for __ in range(env.gridHeight)]

    # display agent switch case (dictionary)
    def all(self, agent):
        return colorRed

    def bySexe(self, agent):
        if agent.getSexe() == female:
            return colorPink
        else:
            return colorBlue

    def byMetabolism(self, agent):
        if agent.getMetabolism() > 2:
            return colorRed
        else:
            return colorBlue

    def byVision(self, agent):
        if agent.getVision() > 3:
            return colorRed
        else:
            return colorBlue

    def byGroup(self, agent):
        #        if bin(agent.getTags()).count('1') > agent.getTagsLength()>>1:
        if agent.getTribe() == 1:
            return colorRed
        else:
            return colorBlue

    agentColorSchemes = {0: all, 1: bySexe, 2: byMetabolism, 3: byVision, 4: byGroup}

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

    # put game update code here
    def updateGame(self):
        # for agents' logs
        metabolism = 0
        vision = 0

        # execute agents randomly
        random.shuffle(self.agents)

        # run agents' rules
        for agent in self.agents:
            # MOVE
            if ruleMoveEat:
                agent.move()
                # remove agent if they're dead
                if agent.getSugar() == 0:
                    # free environment
                    self.env.setAgent(agent.getLocation(), None)
                    # remove or replace agent
                    self.removeAgent(agent)
                    continue

            # COMBAT
            if ruleCombat:
                killed = agent.combat(combatAlpha)
                # if an agent has been killed, remove it
                if killed:
                    # do not free the environment, someone else is already here
                    self.removeAgent(killed)
                # remove agent if they're dead
                if agent.getSugar() == 0:
                    # free environment
                    self.env.setAgent(agent.getLocation(), None)
                    # remove or replace agent
                    self.removeAgent(agent)
                    continue

            # PROCREATE
            if ruleProcreate and agent.isFertile():
                mateItr = agent.mate()
                while True:
                    try:
                        # if a new baby is born, append it to the agents' list
                        self.agents.append(next(mateItr))
                    except StopIteration:
                        break

            # TRANSMIT
            if ruleTransmit:
                agent.transmit()

            # Log agent's parameters
            metabolism += agent.getMetabolism()
            vision += agent.getVision()

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
            self.metabolismMean.append(metabolism / float(numAgents))
            self.visionMean.append(vision / float(numAgents))

        # run environment's rules
        if ruleSeasons:
            S = (self.iteration % (2 * seasonPeriod)) / seasonPeriod
            if S < 1:
                # Summer
                self.season = "(summer, winter)"
                if ruleGrow:
                    self.env.growRegion(northRegion, growFactor1)
                    self.env.growRegion(southRegion, growFactor2)
            else:
                # winter
                self.season = "(winter, summer)"
                if ruleGrow:
                    self.env.growRegion(northRegion, growFactor2)
                    self.env.growRegion(southRegion, growFactor1)
        elif ruleGrow:
            self.season = "NA"
            self.env.grow(growFactor)

        # print("update time: ", round(time.time() - start, 5))

    def draw(self):
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                # display sugar's capacity
                capacity = env.getCapacity((i, j))
                agent = env.getAgent((i, j))
                # change color of site depending on what's on it
                if agent:
                    self.canvas.itemconfig(self.grid[i][j], fill=self.agentColorSchemes[agentColorScheme](self, agent))
                else:
                    self.canvas.itemconfig(self.grid[i][j], fill=colorSugar[capacity - 1] if capacity > 0 else "white")
        self.canvas.pack()


    # put drawing code here
    def initialDraw(self):
        # display Sugarscape
        for i, j in product(range(env.gridHeight), range(env.gridWidth)):
            x1 = 5 + (.5 * self.siteSize) + i * self.siteSize - (.5 * self.siteSize)
            y1 = 5 + (.5 * self.siteSize) + j * self.siteSize - (.5 * self.siteSize)
            x2 = 5 + (.5 * self.siteSize) + i * self.siteSize + (.5 * self.siteSize)
            y2 = 5 + (.5 * self.siteSize) + j * self.siteSize + (.5 * self.siteSize)

            # display sugar's capacity
            capacity = env.getCapacity((i, j))
            agent = env.getAgent((i, j))

            if agent:
                self.grid[i][j] = self.canvas.create_rectangle(x1, y1, x2, y2,
                                             fill=self.agentColorSchemes.get(agentColorScheme)(self, agent),
                                             outline="#C0C0C0")
            else:
                self.grid[i][j] = self.canvas.create_rectangle(x1, y1, x2, y2,
                                             fill=(colorSugar[capacity - 1] if capacity > 0 else "white"),
                                             outline="#C0C0C0")
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
    def mainLoop(self):
        self.update = True
        self.window = Tk()
        self.window.title("Sugarscape")
        self.window.geometry("%dx%d" % (self.width + 5, self.height + 5))
        self.window.resizable(True, True)
        self.window.configure(background='white')
        self.canvas = Canvas(self.window, width=self.width, height=self.height, bg='white')
        last_time = time.time()

        self.window.bind("<Escape>", lambda x: self.setQuit())
        self.window.bind("<F1>", lambda x: self.createPopulationPlot())
        self.window.bind("<F2>", lambda x: self.createWealthPlot())
        self.window.bind("<F3>", lambda x: self.createMetabolismPlot())
        self.window.bind("<F12>", lambda x: self.setPause())
        self.initialDraw()

        while not self.quit:
            # update sugarscape
            if self.update:
                self.updateGame()
                self.iteration += 1

            # display sugarscape state
            self.draw()

            self.window.update()

            if self.popWidget:
                self.popWidget.update(self.population)
            if self.wealthWidget:
                self.wealthWidget.update(self.agents)
            if self.metabolismWidget:
                self.metabolismWidget.update(self.metabolismMean, self.visionMean)

            # calculate and display the framerate
            time_now = time.time()
            time_since_last_frame = time_now - last_time
            framerate = int(round(1.0 / time_since_last_frame, 0))
            last_time = time_now

            # display infos
            if self.update:
                print("Iteration = ", self.iteration, "; fps = ", framerate, "; Seasons (N,S) = ", self.season,
                      "; Population = ", len(self.agents), " -  press F12 to pause.")

    # Generator that formats data in series
    def createFormatSeries(self, xmin, ymin, xmax, ymax, dx, dy, data):
        curve = []
        x = xmin
        for datum in data:
            curve.append(x)
            curve.append(ymin - datum * dy)
            x += dx
            if x >= xmax:
                yield curve
                curve = []
                x = xmin
        yield curve

''' 
Main 
'''

if __name__ == '__main__':

    env = Environment(gridSize)

    # add radial food site 
    env.addFoodSite(northSite, maxCapacity)

    # add radial food site 
    env.addFoodSite(southSite, maxCapacity)

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
    view.mainLoop()
