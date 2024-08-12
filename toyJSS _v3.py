#!/usr/bin/python3
import numpy as np
from itertools import combinations, permutations, repeat
from cpmpy import *

"""
taken from https://developers.google.com/optimization/scheduling/job_shop
each task is labeled by a pair of numbers (m, p) 
where m is the number of the machine the task must be processed on and
p is the processing time of the task

this is a modified example.

We have 
5 jobs {0:4}
3 machines {0:2}
3 operators {0:2}
knowledge level/[operator/machine] {1:3}
preference level/[operator/machine] {1:3}
amounts/[job/machine]

## jobs
job0 = [(100 1)] job0 requires production of 100 parts on machine 2
job1 = [(200 0) (200 2) (200 1)] job1 requires production of 200 parts on machine 1 then 3 then 2
job2 = [(300 1) (300 2)] 
job3 = [(200 1)] 
job4 = [(100 0) (100 2)] 

## operators
op0 = [(2 2 2 3 1 3)] operator 1 has knowledge levels on machines 0,1,2: 2,2,2 and preference levels for machines 0,1,2: 3,1,3 
op1 = [(4 1 1 2 3 2)] 
op2 = [(1 2 3 1 3 1)] 

## execution times on machines per knowledge levels
mach0 = [(20 15 8)] a task on machine 0 takes on average 20 unit times for knowledge level 1, 15 for knowledge level 2, 8 for knowledge level 3
mach1 = [(10 8 6)] 
mach2 = [(10 6 2)]

## Goal: assign jobs to operators so that
1/ experience is maximized
2/ preference is maximized
## Constraints
1/ make span < horizon --> not known here
2/ Precedence constraint:  for any two consecutive tasks in the same job, 
the first must be completed before the second can be started.
3/ No overlap constraints: a machine can't work on two tasks at the same time
4/ No overlap constraints: an operator can't work on two tasks or two machines at the same time
5/ No operator can have no task during e.g. 100 time units
6/ jobs can be interrupted an allocated to another operator.

There are two types of constraints for the job shop problem
1. Precedence constraint:  for any two consecutive tasks in the same job, 
the first must be completed before the second can be started.
2. No overlap constraints: a machine can't work on two tasks at the same time
The objective of the job shop problem is to minimize the makespan: 
the length of time from the earliest start time of the jobs to the latest end time.
"""

###-Dummy data

#sequence_data: matrix J x S with J == number of jobs and S == length of the maximal overall sequence
# row = sequence of machines for job j; if 2 entries are 0, then job needs only 1 machine, 
#                                       if no entries are 0, then the job needs a sequence of 3 machines 
sequence_data = cpm_array([
    [1,-1,-1], 
    [0,2,1],
    [1,2,-1],
    [2,-1,-1],
    [0,2,-1]
    ])
flat_sequence_data = sequence_data.flatten()

#parts_data: vector with J elements == number of jobs; entries are the number of parts to be produced in job j
parts_data = cpm_array([
    5,
    10,
    15,
    10,
    5
    ])



#operator_data: tensor dim operator x machines x level type = 3 x 3 x 2
# operator_data = cpm_array([
#     [[2,3],[2,1],[2,3]], #operator 1: for machine 1 --> knowledge level 2, preference level 3; for machine 2 --> knw 2, pref 1 , ...
#     [[3,2],[1,3],[1,2]], #operator 2: ...
#     [[1,1],[2,3],[3,1]], #operator 3: ...
#     ])
operator_knowl = cpm_array([[2,2,2], 
                  [3,1,1],
                  [1,2,3]])

operator_pref = cpm_array([[3,1,3],
                 [2,3,2],
                 [1,3,1]])

#machining_data: matrix M x E with M number of machines, E levels of experience; entries are the average time spent (per unit) per knowledge level (cols) 
machining_data = cpm_array([
    [8,15,20],
    [6,8,10],
    [2,6,10]
    ])
"""----------------------------------------------------------------------------
some preprocessing
"""
#what is the number of jobs to be executed?
jobs_count = len(parts_data)
range_jobs = range(jobs_count)
#how much machines are available?
machines_count = np.shape(machining_data)[0]
range_machines = range(machines_count)
#how many operators are there?
operators_count = np.shape(machining_data)[0]
range_operators = range(operators_count)
#what is the maximum sequence of machines
max_sequence = np.shape(sequence_data)[1]

#how many parts should be processed on each machine?
binary_sequence = sequence_data
idx = np.where(sequence_data < 0)
idx2 = np.where(sequence_data >= 0)
binary_sequence[idx] = 0
binary_sequence[idx2] = 1


parts_matrix = binary_sequence*parts_data
max_makespan = 0
for m in range(machines_count):
    idx = np.where(sequence_data == m)
    max_makespan += sum(parts_matrix[idx] * machining_data[m])

sum_of_tasks = 0
for i in range(np.shape(parts_matrix)[0]):
    for j in range(np.shape(parts_matrix)[1]):
        sum_of_tasks += parts_matrix[i][j]
range_tasks = range(sum_of_tasks)

#machine to task
flow_machine2task = -1*np.ones(sum_of_tasks)
idxS = 0
idxE = 0
for i in range(0,len(parts_matrix.flatten())):
    if parts_matrix.flatten()[i] != 0:
       idxS = idxE
       idxE += parts_matrix.flatten()[i]
       flow_machine2task[idxS:idxE] = flat_sequence_data[i]
            
flow_machine2task = cpm_array(flow_machine2task)    



"""----------------------------------------------------------------------------
##-decision variables
"""
#here: keep track of each part
#we need a start-time for each part
#start times can overlap, as they link to different machines
# start_time_task = intvar(0, ub_makespan, shape=(1,sum_of_tasks))
# end_time_task = intvar(0, ub_makespan, shape=(1,sum_of_tasks))
# #flow_parts2machines to know which machine is used to exe a task
# flow_tasks2machines = intvar(1, range_machines[-1], shape=(1,sum_of_tasks))
# #flow_parts2operators to know which operator is working on a task
# flow_operators2tasks = intvar(1, range_operators[-1], shape=(1,sum_of_tasks))
# #derive which operator is working on which machine
# #expertise level and preference level can be derived once operator ID is known
# vec_knowledge = intvar(1,4,shape=(1,sum_of_tasks))
# vec_preference = intvar(1,4,shape=(1,sum_of_tasks))


start_time = intvar(0, int(max_makespan), shape=(sum_of_tasks), name="start")
end_time = intvar(0, int(max_makespan), shape=(sum_of_tasks), name="stop")

#flow_machine2task = intvar(0, range_machines[-1], shape=(sum_of_tasks))
flow_operator2task = intvar(0, range_operators[-1], shape=(sum_of_tasks))

knowledge = intvar(1,3,shape=(sum_of_tasks))
preference = intvar(1,3,shape=(sum_of_tasks))
execution_time = intvar(2,20,shape=(sum_of_tasks))


"""----------------------------------------------------------------------------
##-tuning parameters
"""
ub_ergo = 3
lb_allocation4operator = int(round(0.7 * sum_of_tasks/operators_count))
ub_allocation4operator = int(round(1.2 * sum_of_tasks/operators_count))
ub_dur = 100
weight_preference = 1
weight_expertlevel = 1
batch_size = 5
#introduce a scaling factor for the ub of the makespan.
#why? Actual problem has a max makespan of 10 days; this toy example does not have a max makespan
#the scaling factor allows to study the influence of the max makespan on the solution
ub_scaler = 0.5    
ub_makespan = int(ub_scaler * max_makespan)
ub_workscheme = 100
"""----------------------------------------------------------------------------
##-modelling
"""
from itertools import combinations, combinations_with_replacement
from cpmpy import *
from enum import Enum
from functools import total_ordering

model = Model()

###Constraints:

#the first task should start at timestep == 0
model += (min(start_time) == 0)

#the last task should end before the max makespan
model += (max(end_time) <= ub_makespan)

# the end time of a task equals the start time + execution time
# exetime depends on (operator, machine and knowledge level of the operator for the machine)
for i in range_tasks:
    model += (start_time[i] + execution_time[i] == end_time[i])
    for m in range_machines:
        for o in range(0,np.shape(operator_pref)[0]):
            model += (((flow_machine2task[i] == m) & (flow_operator2task[i] == o)).implies(preference[i] == operator_pref[o,m]))
            model += (((flow_machine2task[i] == m) & (flow_operator2task[i] == o)).implies(knowledge[i] == operator_knowl[o,m]))
            model += (((flow_machine2task[i] == m) & (flow_operator2task[i] == o) & (knowledge[i] == operator_knowl[o,m])).implies(
                (execution_time[i] == machining_data[m,(operator_knowl[o,m]-1)])))
            
#each operator should be allocated to a min and max set of tasks
for o in range_operators:
    model += (sum([flow_operator2task[i] == o for i in range_tasks]) >= lb_allocation4operator)
    model += (sum([flow_operator2task[i] == o for i in range_tasks]) <= ub_allocation4operator)
    
goal_knowl = weight_expertlevel * sum(knowledge)
goal_pref = weight_preference * sum(preference)
model.minimize(-goal_knowl-goal_pref)
print('================SOLVING CP====================')
val = model.solve()    

print('#######################################################################################################')
print('Script was validated up to here: PLEASE NOTE THAT CONSTRAINT SET IS INCOMPLETE')
print('#######################################################################################################')

"""
for m in range_machines:
    for i1,i2 in combinations(range_tasks, 2):
        # model += (((flow_machine2task[i1] == m) & (flow_machine2task[i2] == m)).implies(
        #             Xor([(end_time[i1] <= start_time[i2]),(end_time[i2] <= start_time[i1])])))
        if (flow_machine2task[i1] == m) & (flow_machine2task[i2] == m):
            model += (Xor([(end_time[i1] <= start_time[i2]),(end_time[i2] <= start_time[i1])]))


goal_knowl = weight_expertlevel * sum(knowledge)
goal_pref = weight_preference * sum(preference)
model.minimize(-goal_knowl-goal_pref)
#model.minimize(-goal_knowl)
print('================SOLVING CP====================')
val = model.solve()
     

# for i1, i2, i3, i4 in combinations(range_tasks,4):
# # ERGO constraint (simulation): An operator cannot execute more than X tasks on a row on the same machine    
#     model += ((flow_operator2task[i1] == flow_operator2task[i4]).implies(
#              ((flow_machine2task[i1] != flow_machine2task[i2]) |
#              (flow_machine2task[i1] != flow_machine2task[i3]) |
#              (flow_machine2task[i2] != flow_machine2task[i3])))) 
# #reversed ERGO constraint: if operators are the same for 2 tasks, then the times should not equal (start+exe)*4 or the machines should be different
#     model += ((flow_machine2task[i1] == flow_machine2task[i4]).implies(
#              ((flow_operator2task[i1] != flow_operator2task[i2]) |
#              (flow_operator2task[i1] != flow_operator2task[i3]) |
#              (flow_operator2task[i2] != flow_operator2task[i3])))) 


for o1,o2 in combinations(range_tasks,2):
# an operator cannot work on different tasks at the same time
    model += ((flow_operator2task[o1] == flow_operator2task[o2]).implies(
            Xor([end_time[o1]  <= start_time[o2],end_time[o2] <= start_time[o1]])
            ))




#FIRST SEQUENCE: job 2, sequence 0-2-1, parts 10
idxS1 = 0 #start index machine 1
idxE1 = 0 #end index machine 1
idxS2 = 0 #start index machine 2
idxE2 = 0 #end index machine 2
for r in range(0,len(parts_matrix[0,:]),1):
    for c in range(1,len(parts_matrix[0,:]),1):
        idxE1 += parts_matrix[r,c-1]
        idxS1 += idxE1 - parts_matrix[r,c-1]
        if parts_matrix[r,c] > 0: #then there is a sequence
            idxS2 += idxE1
            idxE2 += idxS2 + parts_matrix[r,c]
            model += (start_time[idxS2:idxE2] >= end_time[idxS1:idxE1])
            
#makespan = max(end_time_task)
# model.minimize(makespan)
# model.solve()

#prefered goal: 
### objective

"""


"""weight_preference = 1
weight_expertlevel = 1"""
##objective is to maximize the knowledge level and the preference level
#goal_knowl = weight_expertlevel * sum(knowledge)
#goal_pref = weight_preference * sum(preference)
#model.minimize(-goal_knowl-goal_pref)
print('================SOLVING CP====================')
#val = model.solve()

"""
#print(model.status())
val = model.solve()
print("Makespan:",makespan.value())
print("Schedule:")
grid = -8*np.ones((machines_count, makespan.value()), dtype=int)
for j in all_jobs:
    for m in all_machines:
        grid[m,start_time[m,j].value():end_time[m,j].value()] = j
print(grid)
"""