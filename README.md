# Markov_decision_process

Optimal Maintenance Scheduling for Aircraft Maintenance based on Markov Decision Process. 

Aircraft fleets are composed of older and newer aircraft which requires different maintenance frequencies. Therefore, airlines do not dispose of all their fleet depending on the maintenance. 
Multiple uncertainties intervene in the scheduling maintenance:
- Fuel price 
- number of passengers
The goal is to determine the best policy to minimize the maintenance costs, in other words when to use which aircraft based on the passenger influence estimation and fuel costs stochasticity.

In policy_iteration_15.py, an algorithm is implemented to determine the best maintenance policy with a certain horizon. 
In this file is detailed:
- action_space (maintenance or flight)
- reward_func (action costs)
- transition (is the aircraft operational?)
