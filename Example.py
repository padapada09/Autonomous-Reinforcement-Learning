import learning
import gym
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

states = [0,0,0,0]
actions = [0,0]

maximum = [2.5, 4, 0.8, 4]
minimum = [-2.5, -4, -0.8,-4]

episodes = 10000
Intents = []

cart = learning.Agent(
    states = states,
    sampleSize = [10,1,30,30],
    posibleActions = actions,
    expectedReturn = 205,
    gamma = 1,
    batchSize= 1,
    alpha = "auto",
    epsilon= 0.3,
    learningTime = 202,
    continues = True,
    maximum = maximum,
    minimum = minimum
    )

while (not(cart.ready)):

    observation = env.reset()
    episodeReward = 0
    cart.setState(observation)

    for step in range(200):

        action = cart.decide()

        observation, reward, done, info = env.step(action)

        episodeReward += reward

        cart.learn(observation,action,reward)

        if done:
            cart.update(episodeReward)
            break