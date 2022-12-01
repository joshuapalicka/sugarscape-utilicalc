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
import tkinter as tk

''' 
initial simulation parameters
'''

# view
screenSize = 600, 600
colorBackground = 250, 250, 250

# yellow shades

colorSugar = "#F2FA00"

colorSpice = "#9B4722"

colorBoth = "#BF8232"

colorRed = "#FA3232"

colorPink = "#FA32FA"

colorBlue = "#3232FA"

# environment
gridSize = 50, 50
northSite = 35, 15, 18
southSite = 15, 35, 18

westSite = 10, 10, 14
eastSite = 40, 40, 14

"""
Non-touching circles:

northSite = 38, 12, 12 
southSite = 12, 38, 12

westSite = 12, 12, 12
eastSite = 38, 38, 12

"""

maxCapacity = 10  # !!! < or = nbr items in colorSugar array
seasonPeriod = 50
northRegion = 0, 0, 49, 24
southRegion = 0, 25, 49, 49
growFactor = 1
growFactor1 = 1
growFactor2 = float(growFactor1) / 8

# agents
# agentColorScheme: Agents colour meaning = 0:all, 1:bySex, 2:byMetabolism, 3:byVision, 4:byGroup
maxAgentMetabolism = 4
maxAgentVision = 10
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
    (50, tags0, (0, 50, 0, 50)),  # blues
    (50, tags1, (0, 50, 0, 50))]  # reds
ruleGrow = True
ruleSeasons = False
ruleMoveEat = True
ruleCombat = False
ruleLimitedLife = True
ruleReplacement = False
ruleProcreate = True
ruleTransmit = True
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
    return RGBToHex(tuple(int(c + (255 - c) * (1-factor)) for c in rgb))


def initAgent(agent, tags, distribution):
    newLocation = agent.getEnv().getRandomFreeLocation(distribution)
    if newLocation == None:
        return False
    agent.setLocation(newLocation)
    agent.setMetabolism(random.randint(1, maxAgentMetabolism))
    if ruleSpice:
        agent.setMetabolism(random.randint(1, maxAgentMetabolism), sugar=False)
        agent.setInitialEndowment(random.randint(initEndowment[0], initEndowment[1]), sugar=False)

    agent.setVision(random.randint(1, maxAgentVision))
    agent.setInitialEndowment(random.randint(initEndowment[0], initEndowment[1]))
    agent.setAge(random.randint(minmaxAgentAge[0], minmaxAgentAge[1]))
    sex = random.randint(0, 1)
    agent.setSex(sex)
    agent.setFertility((random.randint(childbearing[sex][0], childbearing[sex][1]),
                        random.randint(childbearing[sex][2], childbearing[sex][3])))
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
        self.grid = [[(None, None) for __ in range(env.gridWidth)] for __ in range(env.gridHeight)]

    # display agent switch case (dictionary)
    def all(self, agent):
        return colorRed

    def bySex(self, agent):
        if agent.getSex() == female:
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
                if ruleSpice:
                    if agent.getSpice() == 0:
                        self.env.setAgent(agent.getLocation(), None)
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
            if ruleSpice:
                self.env.grow(growFactor, sugar=False)
            else:
                self.env.grow(growFactor)

    def draw(self):
        for row in range(len(self.grid)):
            for col in range(len(self.grid[row])):
                # display sugar's capacity
                sugarCapacity = env.getCapacity((row, col))
                agent = env.getAgent((row, col))
                # change color of site depending on what's on it - but only if it wasn't already that color (performance optimization)
                if agent:
                    agent_color = self.agentColorSchemes[agentColorScheme](self, agent)
                    if self.grid[row][col][1] != agent_color:
                        self.canvas.itemconfig(self.grid[row][col][0], fill=agent_color)
                        self.grid[row][col] = (self.grid[row][col][0], agent_color)
                else:
                    if not ruleSpice:
                        fill_color = lightenColor(colorSugar, sugarCapacity)
                        if self.grid[row][col][1] != fill_color:
                            self.canvas.itemconfig(self.grid[row][col][0], fill=fill_color)
                            self.grid[row][col] = (self.grid[row][col][0], fill_color)
                    else:
                        spiceCapacity = env.getCapacity((row, col), sugar=False)
                        if sugarCapacity >= spiceCapacity:
                            fill_color = lightenColor(colorSugar, sugarCapacity)
                        elif sugarCapacity < spiceCapacity:
                            fill_color = lightenColor(colorSpice, spiceCapacity)
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

            # display sugar's capacity
            sugarCapacity = env.getCapacity((row, col))
            agent = env.getAgent((row, col))
            if agent:
                agent_color = self.agentColorSchemes[agentColorScheme](self, agent)
                self.grid[row][col] = (self.canvas.create_rectangle(x1, y1, x2, y2, fill=agent_color, outline="#C0C0C0"), agent_color)
            else:
                if not ruleSpice:
                    fill_color = lightenColor(colorSugar, sugarCapacity)
                    self.grid[row][col] = (self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline="#C0C0C0"), fill_color)
                else:
                    spiceCapacity = env.getCapacity((row, col), sugar=False)
                    if sugarCapacity > 0:
                        fill_color = lightenColor(colorSugar, sugarCapacity)
                    elif spiceCapacity > 0:
                        fill_color = lightenColor(colorSpice, spiceCapacity)
                    else:
                        fill_color = "white"
                    self.grid[row][col] = (self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline="#C0C0C0"), fill_color)
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

            # display sugarscape state
            self.draw()
            self.window.update()

            if self.popWidget:
                self.popWidget.update(self.population)
                if self.iteration >= 900 and self.iteration % 100 == 0:
                    self.popWidget.makeWider()

            if self.wealthWidget:
                self.wealthWidget.update(self.agents)

            if self.metabolismWidget:
                self.metabolismWidget.update(self.metabolismMean, self.visionMean)
                if self.iteration >= 900 and self.iteration % 100 == 0:
                    self.metabolismWidget.makeWider()

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
    env.addSugarSite(northSite, maxCapacity)

    # add radial food site 
    env.addSugarSite(southSite, maxCapacity)

    if ruleSpice:
        # add radial food site
        env.addSpiceSite(eastSite, maxCapacity)

        # add radial food site
        env.addSpiceSite(westSite, maxCapacity)

    # grow to max capacity
    if ruleGrow:
        env.grow(maxCapacity, sugar=True if not ruleSpice else False)

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
