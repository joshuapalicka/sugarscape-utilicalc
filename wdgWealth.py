'''
Created on 2010-05-01

@author: rv
'''
from tkinter import *


class WdgWealth:
    '''
    classdocs
    '''

    def __init__(self, agents, title="Wealth histogram", width=600, height=300):
        '''
        Constructor
        '''
        self.root = Toplevel()
        self.root.protocol("WM_DELETE_WINDOW", self.root.iconify)
        self.root.title(title)
        self.canvas = Canvas(self.root, width=width, height=height)
        self.canvas.pack()
        self.width = width
        self.height = height
        self.numBins = 10
        self.bins = None
        self.update(agents)

    # Display wealth histogram widget
    def update(self, agents):
        self.canvas.delete("all")
        self.bins = [0 for __ in range(self.numBins)]
        maxWealth = 0
        for agent in agents:
            maxWealth = max(agent.getSugar(), maxWealth)

        binRange = maxWealth / (self.numBins - 1)

        for agent in agents:
            bin = int(agent.getSugar() / binRange)
            self.bins[bin] += 1

        incrx = (self.width - 100) / self.numBins
        incry = max((self.height - 100) / max(self.bins), 1)

        y0 = self.height - 20
        self.canvas.create_text(20 + incrx * self.numBins, y0 + 10, text=10 * int(binRange) - 1)
        for index, bin in enumerate(self.bins):
            x = 20 + index * incrx
            y = y0 - bin * incry
            self.canvas.create_rectangle(x, y0, x + incrx, y, fill="red")
            self.canvas.create_text(x + 20, y - 20, text=bin)
            self.canvas.create_text(x, y0 + 10, text=max(index * int(binRange) - 1, 0))

        # self.root.mainloop()
