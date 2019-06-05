import numpy as np
import random

def lastProm(array, amount):
    suma = 0
    last = array.__len__()
    if last == 0:
        return 0
    if last < amount:
        for x in array:
            suma += x
        return suma/(last)
    for x in range(amount):
        suma += array[last-(x+1)]
    return suma/amount

class Agent:

    def __init__(self,state,posibleActions,sampleSize = [],epsilon = 0,alpha = 0,gamma = 0,epsilonDecay = 0,
        alphaDecay = 0, expectedReturn = 0, continues = False, minimum = [], maximum = [], auto = False):
        self.state = state
        self.actions = posibleActions
        self.sampleSize = sampleSize
        self.epsilon = 1 - epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.epsilonDecay = epsilonDecay
        self.alphaDecay = alphaDecay
        self.policy = self.initializePolicy(state,posibleActions,sampleSize)
        self.continues = continues
        self.minimum = minimum
        self.maximum = maximum
        self.record = []
        self.expectedReturn = expectedReturn
        self.alphaOriginal = alpha
        self.epsilonOriginal = epsilon
        if (auto):
            self.memory = []
            self.alpha = random.random()*1
            self.alphaOriginal = self.alpha
            self.epsilon = random.random()*1
            self.epsilonOriginal = self.epsilon
            self.epsilonDecay = random.random()*1
            self.alphaDecay = random.random()*1
        

    def decide(self):
        inputs = self.policy[tuple(self.state)]
        probability0 = np.exp((inputs-np.amax(inputs))*self.epsilon) / float(sum(np.exp((inputs-np.amax(inputs))*self.epsilon)))
        probability1 = np.exp((inputs-np.amax(inputs))*self.epsilon) / float(sum(np.exp((inputs-np.amax(inputs))*self.epsilon)))
        probability1.sort()
        chance = 0
        posibility = random.random()
        for action in probability1:
            chance += action
            if chance > posibility:
                return np.where(probability0 == action)[0][0]

    def update(self,reward):
        if self.record.__len__() < 1:
            self.rewardOffset = abs(reward - self.expectedReturn)
        self.record.append(reward)
        newAlpha = self.alphaOriginal * self.alphaDecay * (1/pow(self.alphaDecay, abs(lastProm(self.record, 200) - self.expectedReturn) / self.rewardOffset))
        newEpsilon = self.epsilonOriginal * self.epsilonDecay * (1/pow(self.epsilonDecay, abs(lastProm(self.record, 200) - self.expectedReturn) / self.rewardOffset))
        self.alpha = newAlpha
        self.epsilon = newEpsilon

    def optimize(self, reward):
        self.memory.append([reward, self.alphaOriginal, self.epsilonOriginal, self.alphaDecay, self.epsilonDecay])
        self.alpha = random.random()
        self.alphaOriginal = self.alpha
        self.epsilon = random.random()
        self.epsilonOriginal = self.epsilon
        self.alphaDecay = random.random()
        self.epsilonDecay = random.random()
        self.policy = self.initializePolicy(self.state,self.actions,self.sampleSize)
        self.record = []

    def learn(self, newState, actionTaken, reward):
        if self.continues:
            newStatef = self.discretize(newState)
        fullState = self.state
        fullState.append(actionTaken)
        old_value = self.policy[tuple(fullState)]
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma*(max(self.policy[tuple(newStatef)])))
        self.policy[tuple(fullState)] = new_value
        self.setState(newState)

    def initializePolicy(self,states,actions,sampleSize):
        array = []
        for state in range(states.__len__()):
            array.append(sampleSize[state])
        array.append(actions.__len__())
        policy = np.zeros(array)
        return policy

    def discretize(self, states):
        j = -1
        discreteState = []
        for state in states:
            j += 1
            increment = (abs(self.minimum[j])+self.maximum[j])/self.sampleSize[j]
            for i in range(0, self.sampleSize[j]):
                if state < increment*(i+1) + self.minimum[j]:
                    discreteState.append(i)
                    break
        return discreteState

    def setState(self,state):
        if self.continues :
            self.state = self.discretize(state)
        if not(self.continues):
            self.state = state