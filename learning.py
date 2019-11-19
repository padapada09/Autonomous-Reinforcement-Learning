from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as Kernels
import matplotlib.pyplot as plt
import numpy as np
import os.path
import json
import codecs

def lastProm(array, amount):
    suma = 0
    last = len(array)
    if last == 0:
        return 0
    if last < amount:
        for x in array:
            suma += x
        return suma/(last)
    for x in range(amount):
        suma += array[last-(x+1)]
    return suma/amount

def product(*args, repeat=1):
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)

class Agent:

    #By default the main 3 hyperparameters will be set as "auto".
    def __init__(self, learningTime, states,posibleActions,sampleSize = [],epsilon = "auto",alpha = "auto",gamma = "auto", epsilonDecay = 1,
        alphaDecay = 1, expectedReturn = 0, batchSize=3, continues = False, minimum = [], maximum = []):
        
        #autoLearners -> Array with every parameter set to "auto".
        self.autoLearners = []

        #record -> Memory for the achived reward in every episode for one combination.
        self.record = []

        #metaRecord -> Memory for the score of every Combination.
        self.metaRecord = []

        #batch -> Here I will record several learning attempts in order to get a proper average.
        self.batch = {'size': batchSize,'attempts':[]}

        #How much time do you think is enough for the learning task given (This hyperparameter should be errased).
        self.learningTime = learningTime

        #Array with one zero for each state the Agent should perceive.
        self.state = states

        #Array with one zero for each action the Agent can take in each state
        self.actions = posibleActions

        #Array with each value been the amount of detail the according state should be break into. 
        self.sampleSize = sampleSize

        #I process the selection of hyperparameter to check for auto requests.
        #   hyperparameters -> Epsilon, Alpha, Gamma.
        self.hyperparameters, self.GPR = self.setAutos({'epsilon':epsilon, 'alpha':alpha, 'gamma':gamma})

        #I keep instances of each hyperparameter so I can give them variations in the learning process while keeping the original value.
        self.epsilon = self.hyperparameters['epsilon']
        self.alpha = self.hyperparameters['alpha']
        self.gamma = self.hyperparameters['gamma']

        #I take the epsilon and alpha decay as the amount they are going to grow. (For example: 1.3 times larger, or 0.5 times the original value).
        self.epsilonDecay = epsilonDecay
        self.alphaDecay = alphaDecay

        #I initialize the policy with all zeros.
        self.policy = self.initializePolicy(states,posibleActions,sampleSize)

        #Check if the enviroment is continues.
        self.continues = continues

        #Take the minimus and maximum of each state-space.
        self.minimum = minimum
        self.maximum = maximum

        #What reward does the Agent has to achive in order to take the learning as resolved.
        self.expectedReturn = expectedReturn

        #True if the learning process is over.
        self.ready = False

    def setAutos(self, parameters):
        
        for parameter, value in parameters.items():

            #I check every parameter given for the "auto" value.
            
            if value == "auto":    

                #If the parameter has the "auto" value, then I give it a random value to start with.
                parameters[parameter] = np.random.random()
                
                #I add the parameters to the list of autoLearners of the class Agent.
                #I have it this way, in order to differentiate the hyperparameters I automattically choose.
                self.autoLearners.append({'name': parameter, 'value' : parameters[parameter], 'boundries' : [0,1]})
        
        kernel = 1.0 * Kernels.RBF() + Kernels.WhiteKernel(noise_level=2)

        #I set the GaussianRegressor here just for convenience. I need to check the reason for each value.
        GaussianRegressor = GaussianProcessRegressor(
            kernel=kernel,
            random_state=1,
            normalize_y=False,
            alpha=0,
            #Not sure how this optimizer works yet.
            n_restarts_optimizer=10)

        return parameters, GaussianRegressor

    def decide(self, greedy = False):
        if (greedy):
            return np.amax(self.policy[tuple(self.state)])
        inputs = self.policy[tuple(self.state)]
        try:
            probability0 = np.exp((inputs-np.amax(inputs))*self.epsilon) / float(sum(np.exp((inputs-np.amax(inputs))*self.epsilon)))
            probability1 = np.exp((inputs-np.amax(inputs))*self.epsilon) / float(sum(np.exp((inputs-np.amax(inputs))*self.epsilon)))
            probability1.sort()
        except:
            print ("The epsilon value is: " + str(self.epsilon))
            exit()
            
        chance = 0
        posibility = np.random.random()
        for action in probability1:
            chance += action
            if chance > posibility:
                return np.where(probability0 == action)[0][0]

    def update(self,reward):
        #Function called every time an episode ends.
        #What I have to do here is:

        #->Record the reward in the "record" attribute. I use the first record when updating alpha and epsilon,
        # so this has to be done before the alpha and epsilon update.
        self.record.append(reward)

        #->Update alpha and epsilon acording to the given decay
        self.alpha = self.hyperparameters['alpha'] * self.alphaDecay * (1/pow(self.alphaDecay, abs(lastProm(self.record, 200) - self.expectedReturn) / abs(self.record[0] - self.expectedReturn)))
        self.epsilon = self.hyperparameters['epsilon'] * self.epsilonDecay * (1/pow(self.epsilonDecay, abs(lastProm(self.record, 200) - self.expectedReturn) / abs(self.record[0] - self.expectedReturn)))

        #->Give the user feedback of the learning process through the console
        print ("Update: " + str(len(self.record)) + " | " + str(self.hyperparameters) +" | " +"Score: " + str(lastProm(self.record,int(self.learningTime/10))),end="\r")

        #->If the learning time is past the 10% of the given by the user:
        if (len(self.record) > (self.learningTime/10) and len(self.record) < (self.learningTime)):
            
            #->Then i check if average of the last 10 scores is under the diagonal line from 0% to 100% of the expectedReturn in the estimated learninTime period.                 
            if (lastProm(self.record,int(self.learningTime/10)) < ((len(self.record)/self.learningTime)*self.expectedReturn)):
                
                #->I set as failure and tell the user the score achived.
                # print ("\nFailure in " + str(len(self.record)) +": ")
                # print ("Score achived: " + str(lastProm(self.record,int(self.learningTime/10))))
                # print ("Changing the configuration...")

                #REMEMBER#
                #I have to use the same configuration of hyperparameters batchSize times,
                # in order to get a good enough aproximation of the score those parameters give.
                
                self.batch['attempts'].append(lastProm(self.record,int(self.learningTime/10)))

                # print (self.batch['attempts']) #Just for debuging

                #As a default I choose the next sampling point as the same I allready tried before, in case the batch is not full.
                nextConfig = {parameter['name'] : parameter['value'] for parameter in  self.autoLearners}

                if (len(self.batch['attempts']) % self.batch['size'] == 0):
                    #If the batch is full, then I have to fit the GaussianProcess with the point:
                    # X = Vector with the Hyperparameter configuration. -> for record in self.metaRecord: record['configuration']
                    # Y = Average score achived in the batch. -> for record in self.metaRecord: record['score']

                    #Default the GPR with a random value selection
                    nextConfig = {parameter['name'] : np.random.random() for parameter in  self.autoLearners}                    
                    
                    #I store the configuration with it's average score from the batch.
                    self.metaRecord.append(self.getMetaData())

                    print ("The last meta record was: " + str(self.metaRecord[len(self.metaRecord)-1])) # Just for debuging

                    #Check if self.metaRecord has more than 2 samples to start using the self.GPR
                    if (len(self.metaRecord) > 2):

                        #Get the data to fit the GPR and then fit it
                        print ("Fitting data")
                        fitting_X_vector = np.array([[config['configuration'][parameter['name']] for parameter in self.autoLearners] for config in self.metaRecord])
                        fitting_Y_vector = np.array([config['score'] for config in self.metaRecord])
                        
                        self.GPR.fit(fitting_X_vector,fitting_Y_vector)
                        #The space has to be given as an array of points such as: [X Y Z] in case of 3 parameter estimation
                        prediction_X_space = np.array(list(product([i/100 for i in range(100)], repeat=len(self.autoLearners))))
                        
                        predicted_mean, uncertainty = self.GPR.predict(prediction_X_space, return_std=True)

                        """
                        Here i should use the gaussian process and an aquisition function to suggest a new point to sample.                        
                        Let's first assume that there's only one auto hyperparamter...
                        Probability of improvement -> pi(x), as the probability of f(x) been greater than the current known bigest value.
                        f -> objective function.
                        f*-> bigest known value. => np.amax(fitting_Y_vector)[0]
                        p -> prediction of f. => predicted_mean
                        u -> uncertainty. => uncertainty
                        pi(x) = P(f(x) > f*) ~ [p(x) + u(x)] - f* + e {'e' is for aproximation}
                        """
                                                
                        ProbabilityOfImprovement = (predicted_mean + uncertainty) - np.amax(fitting_Y_vector)
                        ProbabilityOfImprovement[ProbabilityOfImprovement < 0] = 0
                        if len(self.metaRecord)%10 == 0:
                            plt.plot(ProbabilityOfImprovement,'g')
                            plt.plot(predicted_mean, 'b')
                            plt.fill_between([i for i in range(0,100)],predicted_mean+uncertainty,predicted_mean-uncertainty,color="#b3e8ff")                        
                            plt.scatter([[config['configuration']['alpha']*100] for config in self.metaRecord],[config['score'] for config in self.metaRecord])                        
                            plt.show()
                        
                        # nextConfig = {'alpha' : np.random.random()}
                        # if (nextConfig['epsilon'] <= 0) : nextConfig['epsilon'] = 0.01
                        # Let's try now to be more generic...
                        # nextConfig = {parameter['name'] : np.where(np.amax) for parameter in  self.autoLearners}          
                        otherConfig = {parameter['name'] : np.where(np.amax) for parameter in  self.autoLearners}
                        print (otherConfig)


                self.reset(nextConfig)
        
        if self.record.__len__() > self.learningTime:

            self.endLearning()

    def getMetaData(self):
        parameters = {}
        parameters['configuration'] = {j['name'] : j['value'] for j in self.autoLearners}
        parameters['score'] = lastProm(self.batch['attempts'],self.batch['size'])
        return parameters

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

    def reset(self,newParameters):
        for parameter in self.autoLearners:
            self.hyperparameters[parameter['name']] = newParameters[parameter['name']]
            parameter['value'] = newParameters[parameter['name']]
        self.epsilon = self.hyperparameters['epsilon']
        self.alpha = self.hyperparameters['alpha']
        self.gamma = self.hyperparameters['gamma']
        self.record = []
        self.policy = self.initializePolicy(self.state,self.actions,self.sampleSize)

    def endLearning(self):
        
        jsonPolicy = self.policy.tolist() # nested lists with same data, indices

        file_path = "policy"

        json.dump(jsonPolicy, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

        self.ready = True

        print ("\nThe learning process was succesfull. The policy was store under in the file named 'policy'")