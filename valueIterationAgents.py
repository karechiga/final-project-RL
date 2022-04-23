# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import util
from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code 
        "*** YOUR CODE HERE ***"
        for k in range(self.iterations):
            valuesK = util.Counter()
            for s in self.mdp.getStates():
                action = self.getPolicy(s)
                valuesK[s] = self.getQValue(s,action)
            # update the counter after each iteration
            self.values = valuesK
            
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        if action is None:
            return 0
        tstates = self.mdp.getTransitionStatesAndProbs(state, action)
        q = 0
        for t in tstates:   # sum up each transition state's prob * (reward + discount*value) to get the q value.
            prob = t[1]
            r = self.mdp.getReward(state, action, t[0])
            y = self.discount
            v = self.getValue(t[0])
            q += prob*(r + y*v)
        return q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
        if actions == ():
            return None
        maxQ = -9999999
        act = actions[0]
        for a in actions:   # for each possible action, compute the q value. The highest q value is the action that will be returned.
            q = self.computeQValueFromValues(state, a)
            if q > maxQ:
                maxQ = q
                act = a
        return act

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        # valuesK = util.Counter()
        for k in range(self.iterations):
            s = states[k % len(states)]     # s will cycle through one state per iteration
            action = self.getPolicy(s)
            self.values[s] = self.getQValue(s,action)


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        predecessors = []
        q = util.PriorityQueue()
        for s in states:
            pre = set()    # initialize empty set of predecessors for state s
            for p in states:    # see if p is a predecessor for s
                for action in self.mdp.getPossibleActions(p):
                    for t in self.mdp.getTransitionStatesAndProbs(p, action):
                        if s in t:
                            pre.add(p) # add p to the set of s predecessors
                            break
            predecessors.append(pre)
            # Find the absolute value of the difference between the current value of s
            # in self.values and the highest Q-value across all possible actions from s
            action = self.getPolicy(s)  # highest Q value action
            if not action is None:
                diff = abs(self.values[s] - self.getQValue(s,action))
                q.push(s,-diff)

        for k in range(self.iterations):
            if q.isEmpty():
                return
            s = q.pop()
            action = self.getPolicy(s)
            self.values[s] = self.getQValue(s,action)
            i = states.index(s)
            for p in predecessors[i]:
                action = self.getPolicy(p)  # highest Q value action
                diff = abs(self.values[p] - self.getQValue(p,action))
                if diff > self.theta:
                    q.update(p,-diff)

