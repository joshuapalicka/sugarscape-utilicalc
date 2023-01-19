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
    For each agent (or group of agents equally) affected by the action, the moral value of the action is calculated as follows:
        C * (I * D * N * E) must be calculated for each pleasure and pain, and for each 2nd order pleasure and pain.
        For 2nd order pleasure and pain, the same formula is used, but C is replaced with F or P, respectively.
        Pain is calculated as a negative value, and pleasure is calculated as a positive value.
        The sum of all of these values is the moral value of the action for this/these agent(s).

    All of these values are summed together to get the total moral value of an action.
"""

import math
import random


class Agent:
    """
    This class is used to calculate the moral value of an action on an agent/agents.

    Variables:
    isPleasure = whether the action is a pleasure or pain (if false, output of this consequence is multiplied by -1)
    intensity = the intensity of the action
    duration = the duration of the action
    certainty = the certainty of the action
    propinquity = the propinquity (nearness in time) of the action (1/propinquity^0.1) - arbitrary but needs to decay over time
    multiplier = the multiplier for the number of people affected equally by the action

    Optional Variables:
    f_intensity = the intensity of the 2nd order consequence of the same type
    f_duration = the duration of the 2nd order consequence of the same type
    f_propinquity = the propinquity (nearness in time) of the 2nd order consequence of the same type
    f_multiplier = the multiplier for the number of people affected equally by the 2nd order consequence of the same type

    p_intensity = the intensity of the 2nd order consequence of the opposite type
    p_duration = the duration of the 2nd order consequence of the opposite type
    p_propinquity = the propinquity (nearness in time) of the 2nd order consequence of the opposite type
    p_multiplier = the multiplier for the number of people affected equally by the 2nd order consequence of the opposite type

    Methods:
        getMoralValue() returns the moral value of the action
    """

    def __init__(self, isPleasure, intensity, duration, certainty, propinquity, fecundity, purity, multiplier,
                 f_intensity=0, f_duration=0, f_propinquity=0, f_multiplier=0,
                 p_intensity=0, p_duration=0, p_propinquity=0,
                 p_multiplier=0):

        self.consequences = []
        self.addConsequence(isPleasure, intensity, duration, certainty, propinquity, fecundity, purity, multiplier,
                            f_intensity, f_duration, f_propinquity, f_multiplier, p_intensity, p_duration,
                            p_propinquity,
                            p_multiplier)

    def getConsequences(self):
        return self.consequences

    def addConsequence(self, isPleasure, intensity, duration, certainty, propinquity, fecundity, purity, multiplier,
                       f_intensity=0, f_duration=0, f_propinquity=0, f_multiplier=0,
                       p_intensity=0, p_duration=0, p_propinquity=0,
                       p_multiplier=0):

        adjusted_propinquity = 1 / math.pow(propinquity, .1) if propinquity != 0 else 1

        if isPleasure:
            self.consequences.append(certainty * (intensity * duration * adjusted_propinquity * multiplier))
        else:
            self.consequences.append(-1 * certainty * (intensity * duration * adjusted_propinquity * multiplier))

        # The following lines are used to add 2nd order consequences.
        if isPleasure:
            f_adjusted_propinquity = 1 / math.pow(f_propinquity, .1) if f_propinquity != 0 else 1
            if fecundity != 0:
                self.consequences.append(fecundity * (f_intensity * f_duration * f_adjusted_propinquity * f_multiplier))
            if purity != 0:
                self.consequences.append(-1 * purity * (p_intensity * p_duration * f_adjusted_propinquity * p_multiplier))

        else:
            p_adjusted_propinquity = 1 / math.pow(p_propinquity, .1) if p_propinquity != 0 else 1
            if fecundity != 0:
                self.consequences.append(
                    -1 * fecundity * (f_intensity * f_duration * p_adjusted_propinquity * f_multiplier))
            if purity != 0:
                self.consequences.append(purity * (p_intensity * p_duration * p_adjusted_propinquity * p_multiplier))

    def getMoralValue(self):
        return sum(self.consequences)

    def getNegativeMoralValue(self):
        return sum([x for x in self.consequences if x < 0])


class Decision:
    """
    This class is used to calculate the moral value of a decision on agents.

    Variables:
    self_interest_scale = the scale of self-interest (0 = altruistic, 1 = egoistic)
    agents = a list of agents affected by the decision

    Methods:
        createAgent() creates an agent to add to the decision
        getMoralValue() returns the summed moral value of the decision for all agents
        setSelfInterestScale() sets the self-interest scale
        checkForDecisionMaker() checks if there is a decision maker among the agents
    """

    def __init__(self, selfInterestScale=None):
        self.selfInterestScale = selfInterestScale
        self.agents = []

    def createAgent(self, isPleasure, intensity, duration, certainty, propinquity, fecundity, purity, multiplier,
                    f_intensity=0, f_duration=0, f_propinquity=0, f_multiplier=0,
                    p_intensity=0, p_duration=0, p_propinquity=0, p_multiplier=0, isDecisionMaker=False):

        # if we have a decision maker, we need to note that
        if isDecisionMaker:
            self.agents.append(
                (Agent(isPleasure, intensity, duration, certainty, propinquity, fecundity, purity, multiplier,
                       f_intensity, f_duration, f_propinquity, f_multiplier,
                       p_intensity, p_duration, p_propinquity, p_multiplier), True))

        else:
            self.agents.append(
                (Agent(isPleasure, intensity, duration, certainty, propinquity, fecundity, purity, multiplier,
                       f_intensity, f_duration, f_propinquity, f_multiplier,
                       p_intensity, p_duration, p_propinquity, p_multiplier), False))

    def getAgents(self):
        return self.agents

    def getSelfInterestScale(self):
        return self.selfInterestScale

    def setSelfInterestScale(self, selfInterestScale):
        self.selfInterestScale = selfInterestScale

    def getMoralValue(self):
        totalMoralValue = 0

        for agent in self.agents:
            if self.selfInterestScale is not None:
                if agent[1]:
                    totalMoralValue += agent[0].getMoralValue() * self.selfInterestScale
                else:
                    totalMoralValue += agent[0].getMoralValue() * (1 - self.selfInterestScale)
            else:
                totalMoralValue += agent[0].getMoralValue()

        return round(totalMoralValue, 2)

    def getNegativeMoralValue(self):
        totalMoralValue = 0

        for agent in self.agents:
            if self.selfInterestScale is not None:
                if agent[1]:
                    totalMoralValue += agent[0].getNegativeMoralValue() * self.selfInterestScale
                else:
                    totalMoralValue += agent[0].getNegativeMoralValue() * (1 - self.selfInterestScale)
            else:
                totalMoralValue += agent[0].getNegativeMoralValue()

        return round(totalMoralValue, 2)

    def hasDecisionMaker(self):
        for agent in self.agents:
            if agent[1]:
                return True
        return False


class EvaluateDecisions:
    """
    Evaluates multiple decisions, which are added to this object

    Variables:
    selfInterestScale = the scale of self-interest (0 = altruistic, 1 = egoistic)
    decisions = the decisions to be evaluated

    Methods:
        addDecision() adds a decision to the list of decisions
        setSelfInterestScale() sets the self-interest scale for all decisions
        printMoralValueForAllDecisions() prints the moral value for all decisions
        printBestDecision() prints the decision with the highest moral value
        getBestDecision() returns the name of the decision with the highest moral value
    """

    def __init__(self):
        self.selfInterestScale = None
        self.decisions = []

    def getSelfInterestScale(self):
        return self.selfInterestScale

    def getDecisions(self):
        return self.decisions

    def addDecision(self, decisionName, decision):
        if self.setSelfInterestScale is not None:
            decision.setSelfInterestScale(self.selfInterestScale)
        self.decisions.append((decisionName, decision))
        random.shuffle(self.decisions)  # we randomize this so that, with equal moral values, we choose a random decision

    def setSelfInterestScale(self, selfInterestScale):
        if selfInterestScale > 1 or selfInterestScale < 0:
            raise ValueError("Self interest scale must be between 0 and 1.")

        for decision in self.decisions:
            if not decision[1].hasDecisionMaker():
                print("Warning: Decision " + decision[0] + " does not have a decision maker.")

        if selfInterestScale == 0:
            print("Set self interest to Altruistic")
        elif selfInterestScale == 1:
            print("Set self interest to Egoistic")

        self.selfInterestScale = selfInterestScale
        for decision in self.decisions:
            decision[1].setSelfInterestScale(selfInterestScale)

    def printMoralValueForAllDecisions(self):
        for decision in self.decisions:
            print(decision[0] + ": " + str(decision[1].getMoralValue()))

    def printDecisionWithHighestValue(self):
        bestDecisionName, bestDecision = max(self.decisions, key=lambda x: x[1].getMoralValue())

        print(
            "The best decision from a standard Utilitarian perspective is: " + bestDecisionName + ". with a moral value of " + str(
                bestDecision.getMoralValue()))

        if bestDecision.selfInterestScale is not None:
            print("This decision was made with a self interest scale of " + str(bestDecision.selfInterestScale))
        print()

    # prints the decision with the negative moral value closest to 0
    def printDecisionWithLeastNegativeValue(self):
        bestDecisionName, bestDecision = max(self.decisions, key=lambda x: x[1].getNegativeMoralValue())
        print(
            "The best decision from a Negative Utilitarian perspective is: " + bestDecisionName + ". with a moral value of " + str(
                bestDecision.getMoralValue()))

        if bestDecision.selfInterestScale is not None:
            print("This decision was made with a self interest scale of " + str(bestDecision.selfInterestScale))
        print()

    def getDecisionWithHighestValue(self):
        bestDecisionName, bestDecision = max(self.decisions, key=lambda x: x[1].getMoralValue())
        return bestDecisionName

    def getDecisionWithLeastNegativeValue(self):
        bestDecisionName, bestDecision = max(self.decisions, key=lambda x: x[1].getNegativeMoralValue())
        return bestDecisionName

    def getListOfDecisionsWithHighestValue(self):
        bestDecisionName, bestDecision = max(self.decisions, key=lambda x: x[1].getMoralValue())
        listOfDecisions = []
        for decision in self.decisions:
            if decision[1].getMoralValue() == bestDecision.getMoralValue():
                listOfDecisions.append(decision[0])
        return listOfDecisions
