"""
dynaQTest.py

This script is used to run gridworld several times with different RL algorithms and parameters, and outputs the results.

"""
import valueIterationAgents, qlearningAgents
import gridworld
import random
import matplotlib.pyplot as plt
import numpy as np

def getAgent(opts,mdp):
        
    a = None
    if opts['Agent'] == 'q':
        #simulationFn = lambda agent, state: simulation.GridworldSimulation(agent,state,mdp)
        actionFn = lambda state: mdp.getPossibleActions(state)
        qLearnOpts = {'gamma': opts['Discount'],
                      'alpha': opts['Alpha'],
                      'epsilon': opts['Epsilon'],
                      'actionFn': actionFn}
        a = qlearningAgents.QLearningAgent(**qLearnOpts)
    elif opts['Agent'] == 'dq':    # dyna-Q agent
        actionFn = lambda state: mdp.getPossibleActions(state)
        qLearnOpts = {'gamma': opts['Discount'],
                      'alpha': opts['Alpha'],
                      'epsilon': opts['Epsilon'],
                      'actionFn': actionFn,
                      'pIters': opts['PlanningIters']}
        a = qlearningAgents.DynaQ(**qLearnOpts)
    elif opts['Agent'] == 'dqp':    # Dyna-Q+
        actionFn = lambda state: mdp.getPossibleActions(state)
        qLearnOpts = {'gamma': opts['Discount'],
                      'alpha': opts['Alpha'],
                      'epsilon': opts['Epsilon'],
                      'actionFn': actionFn,
                      'pIters': opts['PlanningIters'],
                      'kappa': opts['Kappa']}
        a = qlearningAgents.DynaQPlus(**qLearnOpts)
    else:
        raise Exception('Unknown agent type: '+opts['Agent'])
    return a

if __name__ == '__main__':
    
    noise = 0 # Noise will always be zero
    epsilon = 0.3 # epsilon (exploration rate) will be consistent
    alpha = 0.3 # alpha (learning rate) will be consistent
    discount = 0.9 # discount will be consistent
    reward = 0 # Living reward will always be consistent
    runs = 100 # number of runs will be consistent.
    episodes = 100 # number of episode iterations
    gridChange = 15 # when the shortcut will be introduced, always consistent
    
    # first run will be on the ShortcutGrid
    g = ['ShortcutGrid', 'MazeGrid'] # grids we are using

    # Agents will be as follows:
    # All agents will have the same noise, epsilon, alpha, discount, reward, and episodes
    # DQ+ agent with PlanningIters = 5, kappa = 0.01
    # DQ agent with PlanningIters = 5
    # Q agent
    agents = [{"Agent": 'dqp',"Noise": noise,"Epsilon": epsilon,"Alpha": alpha,"Discount": discount,"Reward": reward,"Episodes": episodes,"Shortcut": gridChange,"PlanningIters": 5,"Kappa": .01},
              {"Agent": 'dq',"Noise": noise,"Epsilon": epsilon,"Alpha": alpha,"Discount": discount,"Reward": reward,"Episodes": episodes,"Shortcut": gridChange,"PlanningIters": 5,"Kappa": 0},
              {"Agent": 'q',"Noise": noise,"Epsilon": epsilon,"Alpha": alpha,"Discount": discount,"Reward": reward,"Episodes": episodes,"Shortcut": gridChange,"PlanningIters": 0,"Kappa": 0}
              ]
    
    avgCumReturns = [] # average cumulative return across the k episodes for the n times we run
    displayCallback = lambda x: None
    messageCallback = lambda x: None
    pauseCallback = lambda : None
    x = np.arange(1,episodes+1,1) # For plotting the data
    for gridNum in range(len(g)):
        plt.figure(gridNum)
        fig, ax = plt.subplots()
        for agent in agents:
            # For each agent run 100 times
            print()
            print("RUNNING AGENT #", agents.index(agent)+1, " ",agent['Episodes']," EPISODES ", runs," TIMES ON ", g[gridNum])
            print()
            random.seed(10) # Set the same random seed for each agent
            sumReturns = [0]*agent['Episodes'] # initialize with a list of zeros
            for i in range(runs):
                mdpFunction = getattr(gridworld, "get"+g[gridNum])
                mdp = mdpFunction()
                mdp.setLivingReward(reward)
                mdp.setNoise(noise)
                env = gridworld.GridworldEnvironment(mdp)

                a = getAgent(agent,mdp)
                decisionCallback = a.getAction
                
                # RUN EPISODES
                returns = 0
                for episode in range(1, agent['Episodes']+1):
                    # update gridworld at specified episode
                    if len(env.gridWorld.grids) > 1:
                        if episode == agent['Shortcut']:
                            env.gridWorld.updateGrid()
                    returns += gridworld.runEpisode(a, env, agent['Discount'], decisionCallback, displayCallback, messageCallback, pauseCallback, episode)
                    sumReturns[episode-1] += returns
            avgCumReturns.append([x / runs for x in sumReturns]) # gets the average returns for each episode.
            ax.plot(x, avgCumReturns[-1])
        # Plot the data
        fig.legend(['Dyna-Q+','Dyna-Q','Q-Learning'],
                    loc=(0.2,0.6))
        plt.title('Cumulative rewards for Q-Learning methods - ' + g[gridNum])
        plt.xlabel('Number of Episodes')
        plt.ylabel('Average Cumulative Reward')
        plt.savefig('./figures/figure_'+g[gridNum])
    plt.show()