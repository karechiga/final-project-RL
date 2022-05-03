# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from typing import List
from game import *
from learningAgents import ReinforcementAgent
import util
import random


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.qValues = util.Counter() # initialize all qvalues to zero

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.qValues[(state,action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        if len(actions) == 0:
          return 0.0
        # initialize maxQ to -infinity
        maxQ = -99999999
        # check among the legal actions which one has the largest Q Value
        for a in actions:
          q = self.getQValue(state,a)
          if q > maxQ:
            maxQ = q
        return maxQ

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        # initialize maxQ to -infinity
        maxQ = -99999999
        maxAction = None # could be multiple actions that are the same. 
        # using a random tiebreaker to decide which to return.
        # check among the legal actions which one has the largest Q Value
        # return the action
        for a in actions:
          q = self.getQValue(state,a)
          if q > maxQ:
            maxQ = q
            maxAction = [a]
          elif q == maxQ:
            maxAction.append(a)
            # keep maxAction as a list and add to it if there are ties
        return random.choice(maxAction) # random choice tiebreaker

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        # epsilon % of the time, we will take a random action
        # else we will choose the best action
        if util.flipCoin(self.epsilon):
          return random.choice(legalActions)
        action = self.computeActionFromQValues(state)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        self.qValues[(state,action)] = self.qValues[(state,action)] + self.alpha * (reward + self.discount*self.computeValueFromQValues(nextState) - self.qValues[(state,action)])
        return

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

class DynaQ(QLearningAgent):
    """
      Dyna-Q Learning Agent:
      - Inherits from Q-Learning Agent
      - Only difference is that the update function will use
        both simulated experience (using a model) and real experience 
        to learn the values and policy of the state set.
      - NOTE: Environment must be deterministic for this algorithm (i.e. noise = 0)
      
    """
    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)
        self.qValues = util.Counter() # initialize all qvalues to zero
        self.model = {}   # Initialize the model to an empty dictionary
        self.qVisited = set()  # tracks the states and actions that have been visited
    
    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          
          The model will then update from the experience,
          then we will do n iterations of planning using the model.
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        # Updates Q values from real experience first
        self.qValues[(state,action)] = self.qValues[(state,action)] + self.alpha * (reward + self.discount*self.computeValueFromQValues(nextState) - self.qValues[(state,action)])
        # Update the model
        self.model[(state,action)] = (reward,nextState)
        self.qVisited.add((state,action))
        # iterate n times, simulate using the model.
        i = 0
        while i < self.pIters:
          q = random.choice(list(self.qVisited))  # choose a random, already visited state and action
          r,s = self.model[q]
          # Update qValue for q
          self.qValues[q] = self.qValues[q] + self.alpha * (r + self.discount*self.computeValueFromQValues(s) - self.qValues[q])
          i += 1
        return


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)
      
class DynaQPlus(QLearningAgent):
    """
      Dyna-Q+ Learning Agent:
      - Inherits from Q-Learning Agent
      - Same algorithm as Dyna-Q except that it rewards more exploration
      - Keeps track of when a state was last visited. 
      - The longer it has been since it has been visited, the more likely the algorithm will visit again.
      - NOTE: Environment must be deterministic for this algorithm (i.e. noise = 0)
      
    """
    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)
        self.qValues = util.Counter() # initialize all qvalues to zero
        self.model = {}   # Initialize the model to an empty dictionary
        self.qVisited = set()  # tracks the states and actions that have been visited
        
    
    def updateModel(self, state, nextState, action, reward):
      # Contrary to regular Dyna-Q, in Dyna-Q+ we will consider unvisited actions from visited states.
      if state not in self.model.keys():
        self.model[state] = {}
        for a in self.getLegalActions(state):
            # the initial model for such actions was that they would
            # lead back to the same state with a reward of 0.
            if a != action:
                self.model[state][a] = (0, state)
            self.qVisited.add((state,a))
        # add actions to visited q set.

      self.model[state][action] = (reward,nextState)
    
    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          
          The model will then update from the experience,
          then we will do n iterations of planning using the model.
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        # Updates Q values from real experience first
        self.qValues[(state,action)] = self.qValues[(state,action)] + self.alpha * (reward + self.discount*self.computeValueFromQValues(nextState) - self.qValues[(state,action)])
        # Update the model
        self.updateModel(state, nextState, action, reward)
        # iterate n times, simulate using the model.
        self.time += 1
        # increment the timestep by 1, and set the latest state-action time to the current timestep
        self.t[(state,action)] = self.time
        i = 0
        while i < self.pIters:
          q = random.choice(list(self.qVisited))  # choose a random already visited state-action
          r,s = self.model[q[0]][q[1]]
          # Adding additional reward for states that haven't been visited for a while.
          r = r + self.kappa * pow(self.time - self.t[q],0.5)
          # Update qValue for q
          self.qValues[q] = self.qValues[q] + self.alpha * (r + self.discount*self.computeValueFromQValues(s) - self.qValues[q])
          i += 1
        return


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)