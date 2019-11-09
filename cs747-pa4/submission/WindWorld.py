#!/usr/bin/env python
# coding: utf-8
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pdb
class WindWorld:
    def __init__(self, rows, columns, wind,source, destination):
        self.rows = rows
        self.columns = columns
        self.wind = wind
        self.start = source
        self.destination = destination
        
class Agent:
    def __init__(self, world, alpha, epsilon, kingsMoves=False, stochastic = False):
        self.world = world
        self.alpha = alpha
        self.epsilon = epsilon
        if(kingsMoves):
            self.num_actions = 8
        else:
            self.num_actions = 4
        self.stochastic = stochastic
        self.Q = np.zeros((world.rows, world.columns, self.num_actions)) #initializing Q values with zero

    def getNextS(self,currS, a, world):
        if(self.stochastic):
            x = np.random.randint(3)
            if(x==0):
                wind = world.wind
            elif(x==1):
                wind = [i-1 for i in world.wind]
            else:
                wind = [i+1 for i in world.wind]
            
        else:
            wind = world.wind
        i,j = currS
        if(a==0): # go up
            nextS = np.array([i-1-wind[j],j]) # decrease the row
            
        elif(a==1): # go down
            nextS = np.array([i+1-wind[j],j]) # decrease the row
                
        elif(a==2): # go right
            nextS = np.array([i-wind[j],j+1]) # increase the column and subtract the wind
            
        else: # go left
            if(a==3):
                nextS = np.array([i-wind[j],j-1]) # decrease the columns and subtract the wind
            
            else:
                if(self.num_actions==8):
                    if(a==4): # top right
                        nextS = np.array([i-1-wind[j],j+1])
                    
                    elif(a==5): # top left
                        nextS = np.array([i-1-wind[j],j-1])
                        
                    elif(a==6): # bottom right
                        nextS = np.array([i+1-wind[j],j+1])
                        
                    else: # bottom left
                        nextS = np.array([i+1-wind[j],j-1])
            
        # check whether it lies within the boundaries of the world
        if nextS[0] < 0:
            nextS[0] = 0
        if nextS[0] >= world.rows:
            nextS[0] = world.rows - 1 
            
        if nextS[1] < 0:
            nextS[1] = 0
        if nextS[1] >= world.columns:
            nextS[1] = world.columns - 1
        
        return nextS
    
    def qEstimationTD0(self, S,num_episodes):
        num_steps = []
        per_ep = []
        Q = np.zeros((self.world.rows,self.world.columns, self.num_actions))
        # pdb.set_trace()
        steps = 0
        for episode in range(num_episodes):
            x=0
            currS = S
            if(np.random.rand() > self.epsilon):
                currA = np.argmax(Q[currS[0],currS[1]]) # take a greedy action
            else:
                currA = np.random.randint(0,self.num_actions) # take a random action
                
            while(1):
                steps+=1
                x+=1
                nextS = self.getNextS(currS, currA, self.world)
                if(nextS[0]!=self.world.destination[0] or nextS[1]!=self.world.destination[1]): # rewards
                    r = -1
                else:
                    r = 0

                
                if(np.random.rand() > self.epsilon):
                    nextA = np.argmax(Q[nextS[0],nextS[1]])
                else:
                    nextA = np.random.randint(0,self.num_actions) # take a random action

                Q[currS[0],currS[1], currA] += self.alpha * (r + Q[nextS[0],nextS[1], nextA]                                                             - Q[currS[0],currS[1], currA])
                currS = nextS
                currA = nextA

                if r == 0:
                    break
            num_steps.append(steps)
            per_ep.append(x)
        return num_steps,per_ep

# make number of columns equal to the number of elements in the wind
# defining the environment and the agent
def main():
    kingsMoves = False
    stochastic = False
    args = len(sys.argv)
    for i in range(1,args):
        if(sys.argv[i] == "kingsMoves"):
            kingsMoves = True
        if(sys.argv[i] == "stochastic"):
            stochastic = True

    np.random.seed(1)
    rows = 7
    columns = 10
    wind = np.array([0,0,0,1,1,1,2,2,1,0]) # wind has to be subtracted according to the convention
    source = np.array([3,0])
    destination = np.array([3,7])
    epsilon = 0.01
    alpha = 0.5
    num_episodes  = 170
    episodes = [i for i in range(num_episodes)]
    myWorld = WindWorld(rows, columns, wind, source, destination)
    myAgent = Agent(myWorld, alpha, epsilon,kingsMoves, stochastic) # number of actions should be 8 case of kings moves
    time_steps_avg = np.zeros(num_episodes)
    per_episode = np.zeros(num_episodes)
    for i in range(10):
        cum, per = myAgent.qEstimationTD0(source,num_episodes)
        time_steps_avg += cum
        per_episode += per
        # pdb.set_trace()
    time_steps_avg = time_steps_avg/10

    plt.plot(time_steps_avg, episodes)
    plt.ylabel('Episodes')
    plt.xlabel('Time Steps')
    plt.grid()
    plt.show()

    plt.plot(per_episode)
    plt.ylabel('Time Steps per episode')
    plt.xlabel('Episodes')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()