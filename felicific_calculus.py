"""
The purpose of this class is to provide a set of methods for calculating the moral value of an action.
The moral value of an action is the sum of the moral value of the consequences of the action.
We will calculate this by using Jeremy Bentham's formula, called "Felicific Calculus", "Hedonic Calculus", or
"Utilitarian Calculus", which is defined as such:

Variables:
    For pleasure:
    I = the intensity of pleasure
    D = the duration of pleasure
    C = certainty of pleasure
    N = propinquity or nearness in time of the pleasure
    F = fecundity or probability of 2nd order pleasure (the pleasure stemming from the pleasure)
    P = purity or probability of 2nd order pain (the pain stemming from the pleasure)
    E = how many people are affected by the pleasure

    For pain:
    I = the intensity of pain
    D = the duration of pain
    C = certainty of pain
    N = propinquity or nearness in time of the pain
    F = fecundity or probability of 2nd order pain (the pain stemming from the pain)
    P = purity or probability of 2nd order pleasure (the pleasure stemming from the pain)
    E = how many people are affected by the pain

Formula:
    For each actor (or group of actors equally) affected by the action, the moral value of the action is calculated as follows:
        C * (I * D * N * E) must be calculated for each pleasure and pain, and for each 2nd order pleasure and pain.
        For 2nd order pleasure and pain, the same formula is used, but C is replaced with F or P, respectively.
        Pain is calculated as a negative value, and pleasure is calculated as a positive value.
        The sum of all of these values is the moral value of the action for this/these actor(s).

    All of these values are summed together to get the total moral value of an action.
"""

import math
import random


class Actor:
    """
    This class is used to calculate the moral value of an action on an actor/actors.

    Variables:
    isPleasure = whether the action is a pleasure or pain (if false, output of this consequence is multiplied by -1)
    intensity = the intensity of the action
    duration = the duration of the action
    certainty = the certainty of the action
    propinquity = the propinquity (nearness in time) of the action (1/propinquity^0.1) - arbitrary but needs to decay over time
    extent = the extent for the number of people affected equally by the action

    Optional Variables:
    f_intensity = the intensity of the 2nd order consequence of the same type
    f_duration = the duration of the 2nd order consequence of the same type
    f_propinquity = the propinquity (nearness in time) of the 2nd order consequence of the same type
    f_extent = the extent for the number of people affected equally by the 2nd order consequence of the same type

    p_intensity = the intensity of the 2nd order consequence of the opposite type
    p_duration = the duration of the 2nd order consequence of the opposite type
    p_propinquity = the propinquity (nearness in time) of the 2nd order consequence of the opposite type
    p_extent = the extent for the number of people affected equally by the 2nd order consequence of the opposite type

    Methods:
        getMoralValue() returns the moral value of the action
    """

    def __init__(self, isPleasure, intensity, duration, certainty, propinquity, fecundity, purity, extent,
                 f_intensity=0, f_duration=0, f_propinquity=0, f_extent=0,
                 p_intensity=0, p_duration=0, p_propinquity=0,
                 p_extent=0):

        self.consequences = []
        self.addConsequence(isPleasure, intensity, duration, certainty, propinquity, fecundity, purity, extent,
                            f_intensity, f_duration, f_propinquity, f_extent, p_intensity, p_duration,
                            p_propinquity,
                            p_extent)

    def getConsequences(self):
        return self.consequences

    def addConsequence(self, isPleasure, intensity, duration, certainty, propinquity, fecundity, purity, extent,
                       f_intensity=0, f_duration=0, f_propinquity=0, f_extent=0,
                       p_intensity=0, p_duration=0, p_propinquity=0,
                       p_extent=0):

        adjusted_propinquity = 1 / math.pow(propinquity, .1) if propinquity != 0 else 1

        if isPleasure:
            self.consequences.append(certainty * (intensity * duration * adjusted_propinquity * extent))
        else:
            self.consequences.append(-1 * certainty * (intensity * duration * adjusted_propinquity * extent))

        # The following lines are used to add 2nd order consequences.
        if isPleasure:
            f_adjusted_propinquity = 1 / math.pow(f_propinquity, .1) if f_propinquity != 0 else 1
            if fecundity != 0:
                self.consequences.append(fecundity * (f_intensity * f_duration * f_adjusted_propinquity * f_extent))
            if purity != 0:
                self.consequences.append(-1 * purity * (p_intensity * p_duration * f_adjusted_propinquity * p_extent))

        else:
            p_adjusted_propinquity = 1 / math.pow(p_propinquity, .1) if p_propinquity != 0 else 1
            if fecundity != 0:
                self.consequences.append(
                    -1 * fecundity * (f_intensity * f_duration * p_adjusted_propinquity * f_extent))
            if purity != 0:
                self.consequences.append(purity * (p_intensity * p_duration * p_adjusted_propinquity * p_extent))

    def getMoralValue(self):
        return sum(self.consequences)

    def getNegativeMoralValue(self):
        return sum([x for x in self.consequences if x < 0])


class Act:
    """
    This class is used to calculate the moral value of an act on actors.

    Variables:
    self_interest_scale = the scale of self-interest (0 = altruistic, 1 = egoistic)
    actors = a list of actors affected by the act

    Methods:
        createActor() creates an actor to add to the act
        getMoralValue() returns the summed moral value of the act for all actors
        setSelfInterestScale() sets the self-interest scale
        checkForActMaker() checks if there is a act maker among the actors
    """

    def __init__(self, selfInterestScale=None):
        self.selfInterestScale = selfInterestScale
        self.actors = []

    def createActor(self, isPleasure, intensity, duration, certainty, propinquity, fecundity, purity, extent,
                    f_intensity=0, f_duration=0, f_propinquity=0, f_extent=0,
                    p_intensity=0, p_duration=0, p_propinquity=0, p_extent=0, isDecisionMaker=False):

        self.actors.append(
            (Actor(isPleasure, intensity, duration, certainty, propinquity, fecundity, purity, extent,
                   f_intensity, f_duration, f_propinquity, f_extent,
                   p_intensity, p_duration, p_propinquity, p_extent), True if isDecisionMaker else False))

    def getActors(self):
        return self.actors

    def getSelfInterestScale(self):
        return self.selfInterestScale

    def setSelfInterestScale(self, selfInterestScale):
        self.selfInterestScale = selfInterestScale

    def getMoralValue(self):
        totalMoralValue = 0

        for actor in self.actors:
            if self.selfInterestScale is not None:
                if actor[1]:
                    totalMoralValue += actor[0].getMoralValue() * self.selfInterestScale
                else:
                    totalMoralValue += actor[0].getMoralValue() * (1 - self.selfInterestScale)
            else:
                totalMoralValue += actor[0].getMoralValue()

        return totalMoralValue

    def getNegativeMoralValue(self):
        totalMoralValue = 0

        for actor in self.actors:
            if self.selfInterestScale is not None:
                if actor[1]:
                    totalMoralValue += actor[0].getNegativeMoralValue() * self.selfInterestScale
                else:
                    totalMoralValue += actor[0].getNegativeMoralValue() * (1 - self.selfInterestScale)
            else:
                totalMoralValue += actor[0].getNegativeMoralValue()

        return totalMoralValue

    def hasActMaker(self):
        for actor in self.actors:
            if actor[1]:
                return True
        return False


class ActionSelector:
    """
    Evaluates multiple acts, which are added to this object

    Variables:
    selfInterestScale = the scale of self-interest (0 = altruistic, 1 = egoistic)
    acts = the acts to be evaluated

    Methods:
        addAct() adds a act to the list of acts
        setSelfInterestScale() sets the self-interest scale for all acts
        printMoralValueForAllActs() prints the moral value for all acts
        printBestAct() prints the act with the highest moral value
        getBestAct() returns the name of the act with the highest moral value
    """

    def __init__(self):
        self.selfInterestScale = None
        self.acts = []

    def getSelfInterestScale(self):
        return self.selfInterestScale

    def getActs(self):
        return self.acts

    def addAct(self, actName, act):
        if self.setSelfInterestScale is not None:
            act.setSelfInterestScale(self.selfInterestScale)
        self.acts.append((actName, act))
        random.shuffle(
            self.acts)  # we randomize this so that, with equal moral values, we choose a random act

    def setSelfInterestScale(self, selfInterestScale):
        if selfInterestScale > 1 or selfInterestScale < 0:
            raise ValueError("Self interest scale must be between 0 and 1.")

        for act in self.acts:
            if not act[1].hasActMaker():
                print("Warning: Act " + act[0] + " does not have a act maker.")

        self.selfInterestScale = selfInterestScale
        for act in self.acts:
            act[1].setSelfInterestScale(selfInterestScale)

    def printMoralValueForAllActs(self):
        for act in self.acts:
            print(act[0] + ": " + str(act[1].getMoralValue()))

    def printActWithHighestValue(self):
        bestActName, bestAct = max(self.acts, key=lambda x: x[1].getMoralValue())

        print(
            "The best act from a standard Utilitarian perspective is: " + bestActName + ". with a moral value of " + str(
                bestAct.getMoralValue()))

        if bestAct.selfInterestScale is not None:
            print("This act was made with a self interest scale of " + str(bestAct.selfInterestScale))
        print()

    # prints the act with the negative moral value closest to 0
    def printActWithLeastNegativeValue(self):
        bestActName, bestAct = max(self.acts, key=lambda x: x[1].getNegativeMoralValue())
        print(
            "The best act from a Negative Utilitarian perspective is: " + bestActName + ". with a moral value of " + str(
                bestAct.getMoralValue()))

        if bestAct.selfInterestScale is not None:
            print("This act was made with a self interest scale of " + str(bestAct.selfInterestScale))
        print()

    def getActWithHighestValue(self):
        bestActName, bestAct = max(self.acts, key=lambda x: x[1].getMoralValue())
        return bestActName

    def getActWithLeastNegativeValue(self):
        bestActName, bestAct = max(self.acts, key=lambda x: x[1].getNegativeMoralValue())
        return bestActName

    def getListOfActsWithHighestValue(self):
        bestActName, bestAct = max(self.acts, key=lambda x: x[1].getMoralValue())
        listOfActs = []
        for act in self.acts:
            if act[1].getMoralValue() == bestAct.getMoralValue():
                listOfActs.append(act[0])
        return listOfActs

    def getListOfActsWithLeastNegativeValue(self):
        bestActName, bestAct = max(self.acts, key=lambda x: x[1].getNegativeMoralValue())
        listOfActs = []
        for act in self.acts:
            if act[1].getNegativeMoralValue() == bestAct.getNegativeMoralValue():
                listOfActs.append(act[0])
        return listOfActs
