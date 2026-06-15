#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Programming 
Practical for course 'Symbolic AI'
2020, Leiden University, The Netherlands
By Thomas Moerland
"""
import numpy as np
from world import World

class Dynamic_Programming:
    
    def __init__(self):
        self.V_s = None # will store a potential value solution table
        self.Q_sa = None # will store a potential action-value solution table
        
    def value_iteration(self,env,gamma = 1.0, theta=0.001):
        ''' Executes value iteration on env. 
        gamma is the discount factor of the MDP
        theta is the acceptance threshold for convergence '''
        
        print("Starting Value Iteration (VI)")
        # initialize value table
        V_s = np.zeros(env.n_states)
    
        
        
        while True:#zolang er nog geen convergentie is
            DELTA = 0 #hulpvariabele om later convergentie mee te checken
            for s in env.states: #update voor elke state
                x = V_s[s] #controle value table
                k = float('-inf') # hulpvariable om de max die per state kan worden bereikt in op te slaan
                for a in env.actions: #ga alle acties van een state na
                    next_state, reward = env.transition_function(s,a)#bepaal s' en de reward die de agent krijgt als het naar s' gaat
                    b = (reward + gamma * V_s[next_state])#bepaal de voorlopige waarde van V_s[s]
                   
                    if b>k: #update k telkens naar de grootst mogelijke waarde van V_s[s]
                        k = b
         
                V_s[s] = k   #vul de uiteindelijk grootst mogelijke waarde in in V_s[s]       
                DELTA = max(DELTA, np.abs(x - V_s[s]))#update de max error         
            if DELTA < theta:#convergentie check
                break#er is convergentie
            print(DELTA)#print per iteratie de max error
        print(V_s)#print de value table      
        
        d = {1:"A",2:"B",3:"C"}
        d.clear()
        print(d)
        self.V_s = V_s
        return

    def Q_value_iteration(self,env,gamma = 1.0, theta=0.001):
        ''' Executes Q-value iteration on env. 
        gamma is the discount factor of the MDP
        theta is the acceptance threshold for convergence '''

        print("Starting Q-value Iteration (QI)")
        # initialize state-action value table
        Q_sa = np.zeros([env.n_states,env.n_actions])

        
        while True:#zolang er nog geen convergentie is
            DELTA = 0#hulpvariabele om later convergentie mee te checken
            for s in env.states:#update voor elke state
                for a in env.actions:#geef elke actie een kolomwaarde in de state-action value tabel
                    if a == 'up':
                        m = 0
                    if a == 'down':
                        m = 1
                    if a == 'left':
                        m = 2
                    if a == 'right':
                        m = 3


                    x = Q_sa[s][m]#controle value table
                    next_state, reward = env.transition_function(s,a)#bepaal s' en de reward die de agent krijgt als het naar s' gaat
                    
                    k = np.amax(Q_sa, axis = 1)#bepaal voor elke state de max value
                    Q_sa[s][m]= reward + gamma*k[next_state]#Q_sa wordt de reward + de max value van s'

                    DELTA = max(DELTA, np.abs(x - Q_sa[s][m]))#update de max error

            if DELTA < theta:#convergentie check
                break#er is convergentie
            print(DELTA)#print per iteratie de max error
        print(Q_sa)#print de state-action value table
        

        self.Q_sa = Q_sa
        return
                
    def execute_policy(self,env,table='V'):
        ## Execute the greedy action, starting from the initial state
        env.reset_agent()
        print("Start executing. Current map:") 
        env.print_map()
        while not env.terminal:
            current_state = env.get_current_state() # this is the current state of the environment, from which you will act
            available_actions = env.actions
            # Compute action values
            if table == 'V' and self.V_s is not None:
                
                b = float('-inf')#hulpvariabele om later de beste actie te doen met de hoogste value


                for a in available_actions: #kijk voor alle acties
                    next_state,reward = env.transition_function(current_state,a)#bepaal s' en de reward die de agent krijgt als het naar s' gaat
                    k = self.V_s[next_state] #k wordt de value van de volgende state
                    print(a,k)


                    if k>b:#als er een actie wordt gevonden met een hogere value van de volgende state
                        b = k #dan updaten we de max value naar deze hogere value
                        z = a #en wordt ook de uit te voeren actie geupdatet

                    if  reward==self.V_s[current_state]:#als de reward van de actie gelijk is aan de value van de huidige state
                        z = a # dan voeren we deze actie uit
                        break


                greedy_action = z 

                
            
            elif table == 'Q' and self.Q_sa is not None:
                
                b= float('-inf')#hulpvariabele om de actie met de hoogste value te doen
                for x in range(0, 4):#kijk voor alle mogelijke acties
                    k = self.Q_sa[current_state][x]#de value als het vanaf de huidige state actie x uitvoert

                    if k>b:#als deze actie een hogere value oplevert
                        b = k#dan updaten we deze hogere value 
                        z = x#en wordt ook de uit te voeren actie geupdatet
                #converteer een actie van int naar string, zodat het kan worden uitgevoerd
                if z == 0:
                    m = 'up'
                if z == 1:
                    m = 'down'
                if z == 2:
                    m = 'left'
                if z == 3:
                    m = 'right'

                greedy_action = m 
                
                
            else:
                print("No optimal value table was detected. Only manual execution possible.")
                greedy_action = None


            # ask the user what he/she wants
            while True:
                if greedy_action is not None:
                    print('Greedy action= {}'.format(greedy_action))    
                    your_choice = input('Choose an action by typing it in full, then hit enter. Just hit enter to execute the greedy action:')
                else:
                    your_choice = input('Choose an action by typing it in full, then hit enter. Available are {}'.format(env.actions))
                    
                if your_choice == "" and greedy_action is not None:
                    executed_action = greedy_action
                    env.act(executed_action)
                    break
                else:
                    try:
                        executed_action = your_choice
                        env.act(executed_action)
                        break
                    except:
                        print('{} is not a valid action. Available actions are {}. Try again'.format(your_choice,env.actions))
            print("Executed action: {}".format(executed_action))
            print("--------------------------------------\nNew map:")
            env.print_map()
        print("Found the goal! Exiting \n ...................................................................... ")
    

def get_greedy_index(action_values):
    ''' Own variant of np.argmax, since np.argmax only returns the first occurence of the max. 
    Optional to uses '''
    return np.where(action_values == np.max(action_values))
    
if __name__ == '__main__':
    env = World('prison.txt') 
    DP = Dynamic_Programming()

    # Run value iteration
    input('Press enter to run value iteration')
    optimal_V_s = DP.value_iteration(env)
    input('Press enter to start execution of optimal policy according to V')
    DP.execute_policy(env, table='V') # execute the optimal policy
    
    # Once again with Q-values:
    input('Press enter to run Q-value iteration')
    optimal_Q_sa = DP.Q_value_iteration(env)
    input('Press enter to start execution of optimal policy according to Q')
    DP.execute_policy(env, table='Q') # execute the optimal policy

