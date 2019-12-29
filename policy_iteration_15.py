# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:13:46 2019

@author: eleon
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:42:34 2019

@author: eleon
"""

import numpy as np
from passengers import loadfactor_monthly
from passengers import get_sample 
from passengers import fuel
import matplotlib.pyplot as plt
import math
import numbers
import itertools
#--------------------Problem Parameters
#d=8 # Horizon(in months);
gamma=1;
capacity=130;
capacity_1=150;
capacity_2=120;
year0=2020;
state_space=np.array([[1,1],[1,0],[0,1],[0,0]])
#Initial conditions and problem parameter declaration
#----------------------------Parameters that will remain fixed during time stepping

#Operating cost per passenger:

OC=np.array([[40,10.2,30],
    [40,15,40]]) # OC[0]: fixed costs independent of aircraft(like terminal/port costs)
                       # OC[1]: fuel cost per passenger per litre. Needs to be * by total fuel volume used
                         # OC[2]: fixed costs dependent on aircraft(maintenance etc)
#Ticket Price:
ticket_price_1=110   # Flat, independent of plane
ticket_price_2=150

#-----------------------------Parameters that will vary during timestepping in MDP

#---Availability Probabilities
#Store pseudocounts of beta distribution for availability of both aircraft


#---Load Factor

[mean_values,std_values]=loadfactor_monthly() # mean_values consist of a 12x2 matrix, 12: monthly and 2: parameters m and c of linear fit y=mx+c 
                                            # where y is the predicted  mean of load factor in the desired year for that month. std is the stan
                                            # dard deviation of load factor in that month(predicted).

#---Current Time Parameters

curr_month=2
curr_year=2020

#-----
#---Fuel Cost
#fuel_ppl=1.78  # This needs to change, introduce some stochasticity
#fuel_price=1.78

#state=[np.random.beta(counts_newplane),np.random.beta(counts_oldplane)];
#state=[1,1]
#actions=action_space(state);
#[exp_profit,policy]=best_action(counts_newplane,counts_oldplane,mean_values,std_values,curr_month,curr_year,d,fuel_price,state);




def action_space(state): #Possible actions from given state

    if state[0]==1:

        if state[1]==1:
            return [1,2];
            
        if state[1]==0:
            return [1];
        
    if state[0]==0:

        if state[1]==1:
            return [2];

        if state[1]==0:
            return [0];

def reward_func(OC_current,curr_month,curr_year,load_factor): # Reward function: Use as it is

    global capacity_1
    global capacity_2
    global ticket_price_1
    global ticket_price_2
    global year0
    
    fuel_ppl=fuel(curr_month,curr_year-year0)*8.6301
    
    if load_factor >0.85:
        ticket_price=ticket_price_2;
    else:
        ticket_price=ticket_price_1;
        
    operating_cost= OC_current[0]+OC_current[1]*fuel_ppl+OC_current[2];
    
    if OC_current[2]==30:
        return capacity_1*(ticket_price*load_factor-operating_cost);
    else:
        return capacity_2*(ticket_price*load_factor-operating_cost);
    
def update_counts(i_action,counts_newplane,counts_oldplane):  # This function updates Beta dist. counters based on actions
    
    if i_action ==1:

        counts_newplane_new=[counts_newplane[0],counts_newplane[1]+1];
        counts_oldplane_new=[counts_oldplane[0]+2,counts_oldplane[1]];
        return [counts_newplane_new,counts_oldplane_new];
    
    if i_action==2:

        counts_oldplane_new=[counts_oldplane[0],counts_oldplane[1]+1];
        counts_newplane_new=[counts_newplane[0]+2,counts_newplane[1]];
        return [counts_newplane_new,counts_oldplane_new];

    if i_action==0:
        
        counts_newplane_new=[counts_newplane[0]+2,counts_newplane[1]];
        counts_oldplane_new=[counts_oldplane[0]+2,counts_oldplane[1]];
        return [counts_newplane_new,counts_oldplane_new];


def transition(counts_newplane,counts_oldplane,state): # Calculates T(s'|s,a)

    ticker_newplane=0.01;
    ticker_oldplane=0.01;

    if counts_newplane[0]-counts_newplane[1]>4:
        ticker_newplane=0.99;
    if counts_oldplane[0]-counts_oldplane[1]>4:
        ticker_oldplane=0.99;
        
    if state[0]==1:
        if state[1]==1:
            return ticker_newplane*ticker_oldplane
        if state[1]==0:
            return ticker_newplane*(1-ticker_oldplane);
        
    if state[0]==0:
        if state[1]==1:
            return (1-ticker_newplane)*ticker_oldplane
        if state[1]==0:
            return (1-ticker_newplane)*(1-ticker_oldplane)
        
        
def noaction_reward(): # Reward for reaching state [0,0] and hence taking no 
    
    global OC
    global capacity
    return -OC[0,0]*capacity


def execute_policy(policy,i_month,counts_newplane,counts_oldplane,mean_values,std_values,curr_month,curr_year,state):
    # This function is meant to simulate for an arbitrary policy, as shown below
    actions=policy[i_month];              
    
    if actions==0:   # If no action can be taken from current state, since both planes are unavailable
        
        reward_current=noaction_reward();
        [counts_newplane_new,counts_oldplane_new]=update_counts(actions,counts_newplane,counts_oldplane)
        expected_prob=np.zeros([state_space.shape[0]]);
        
        count_istate=0;
        
        for i_state in state_space:
            expected_prob[count_istate]=transition(counts_newplane_new,counts_oldplane_new,i_state);
            count_istate+=1
            
        temp=np.random.choice(range(state_space.shape[0]),p=expected_prob); # Simulate next state according to probabilities
        next_state=state_space[temp,:];
        
    else:

        load_factor=(0.01)*np.random.normal(curr_year*mean_values[curr_month-1,0]+mean_values[curr_month-1,1],std_values[curr_month-1])
        OC_current=OC[actions-1,:];
        reward_current=reward_func(OC_current,curr_month,curr_year,load_factor)
        [counts_newplane_new,counts_oldplane_new]=update_counts(actions,counts_newplane,counts_oldplane)
        
        expected_prob=np.zeros([state_space.shape[0]]);
        count_istate=0;
        
        for i_state in state_space:
            expected_prob[count_istate]=transition(counts_newplane_new,counts_oldplane_new,i_state);
            count_istate+=1;

        temp=np.random.choice(range(state_space.shape[0]),p=expected_prob); # Simulate next state according to probabilities
        next_state=state_space[temp,:];
        
    curr_month+=1;
    if curr_month==13:
        curr_month=1;
        curr_year+=1;
        
    return [reward_current,counts_newplane_new,counts_oldplane_new,next_state,curr_month,curr_year];

def compute_policy(policy,d,counts_newplane,counts_oldplane,mean_values,std_values,curr_month,curr_year,state):
    total_reward=0
    #print('init',state)
    for i_month in range(d):
        switch_a=0
        
        possible_actions=action_space(state);
        
        if ((policy[i_month] in possible_actions)) : 
            switch_a=1;
        
        if switch_a==0: # action not possible when we are at this state 
            #print(i_month,policy[i_month])
            return -float('inf')
        
        [reward_current,counts_newplane_new,counts_oldplane_new,next_state,next_month,next_year]=execute_policy(policy,i_month,counts_newplane,counts_oldplane,mean_values,std_values,curr_month,curr_year,state)
        
        total_reward+=reward_current;
        counts_newplane=counts_newplane_new;
        counts_oldplane=counts_oldplane_new;
        
        state=next_state
        #print(i_month,state)
        curr_month=next_month;
        curr_year=next_year;
        
    return total_reward


def policies(d):
    '''
    function that takes the horizon and return all the possible policies
    '''
    return [i for i in itertools.product(np.array([0,1,2]),repeat=d)]
    

def policy_iteration(d):
    global curr_month
    global curr_year

    state=[1,1]
    counts_newplane=[4,1];
    counts_oldplane=[4,1];    
    reward=[]
    
    for policy in policies(d):
        reward_current=compute_policy(policy,d,counts_newplane,counts_oldplane,mean_values,std_values,curr_month,curr_year,state)
        reward.append(reward_current)
        
    idx=reward.index(max(reward))
    best_reward=reward[idx]
    best_policy=policies(d)[idx]
    
    #for i in range(len(policies(d))):
        #if reward[i]!=-float('inf'):
            #print(policies(d)[i],reward[i])
    return best_policy, best_reward

#---------------------------------- Greedy Policy -------------------------------------------------------------
                    
def greedy_policy(d):
    curr_year=2020;
    curr_month=2;
    total_reward=0;
    counts_newplane=[4,1];
    counts_oldplane=[4,1];
    state=[1,1];
    [mean_values,std_values]=loadfactor_monthly()
    
    total_reward=0
    for i_month in range(d):      
        [reward_current,counts_newplane_new,counts_oldplane_new,next_state,next_month,next_year]=execute_policy_2(1,counts_newplane,counts_oldplane,mean_values,std_values,curr_month,curr_year,d,state)
        total_reward+=reward_current;
        counts_newplane=counts_newplane_new;
        counts_oldplane=counts_oldplane_new;
        state=next_state;
        curr_month=next_month;
        curr_year=next_year;
    
    return total_reward


def execute_policy_2(actions,counts_newplane,counts_oldplane,mean_values,std_values,curr_month,curr_year,d,state):
    switch_a=0;                             # This function is meant to simulate for an arbitrary policy, as shown below
    
    possible_actions=action_space(state);
    if ((actions in possible_actions)) :  # Policy #1: 1 or nothing
        switch_a=1;
    
    elif (actions+1 in possible_actions):  # Comment out for  1 or nothing policy
        actions+=1;
        switch_a=1;                      #Policy #2: 1 when possible, if 1 not available 2.
    
    if  switch_a==0: # If no action can be taken from current state, since both planes are unavailable
        reward_current=noaction_reward();
        i_action=0
        [counts_newplane_new,counts_oldplane_new]=update_counts(i_action,counts_newplane,counts_oldplane)
        expected_prob=np.zeros([state_space.shape[0]]);
        count_istate=0;
        
        for i_state in state_space:
            expected_prob[count_istate]=transition(counts_newplane_new,counts_oldplane_new,i_state);
            count_istate+=1
        temp=np.random.choice(range(state_space.shape[0]),p=expected_prob); # Simulate next state according to probabilities
        next_state=state_space[temp,:];
        
    else:
        i_action=actions;
        load_factor=(0.01)*np.random.normal(curr_year*mean_values[curr_month-1,0]+mean_values[curr_month-1,1],std_values[curr_month-1])
        OC_current=OC[i_action-1,:];
        reward_current=reward_func(OC_current,curr_month,curr_year,load_factor)
        [counts_newplane_new,counts_oldplane_new]=update_counts(i_action,counts_newplane,counts_oldplane)
        expected_prob=np.zeros([state_space.shape[0]]);
        count_istate=0;
        for i_state in state_space:
            expected_prob[count_istate]=transition(counts_newplane_new,counts_oldplane_new,i_state);
            count_istate+=1;
        temp=np.random.choice(range(state_space.shape[0]),p=expected_prob); # Simulate next state according to probabilities
        next_state=state_space[temp,:];
    curr_month+=1;
    
    if curr_month==13:
        curr_month=1;
        curr_year+=1;
    return [reward_current,counts_newplane_new,counts_oldplane_new,next_state,curr_month,curr_year];

# ------------------------------------------ Simulations --------------------------------------------

n=1000
d=10

best_policies=[]
greedy_policies=[]
average_best=[]
average_greedy=[]

for i in range(n):
    best_policies.append([policy_iteration(d)])
    greedy_policies.append(greedy_policy(d))
    reward=np.array(best_policies)[:,0,1]
    reward_greedy=np.array(greedy_policies)
    average_best.append(sum(reward)/(i+1))
    average_greedy.append(sum(reward_greedy)/(i+1))


    
#reward=np.array(best_policies)[:,0,1]
#reward_greedy=np.array(greedy_policies)

x=np.linspace(1,n,n)

fig, ax = plt.subplots()
ax.plot(np.arange(len(average_best)),average_best*30,label='policy iteration')
ax.plot(np.arange(len(average_greedy)),average_greedy*30,label='greedy policy')
ax.legend()


plt.show()