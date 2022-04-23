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
