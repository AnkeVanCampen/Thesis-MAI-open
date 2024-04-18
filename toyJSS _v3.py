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
    [2,0,0], 
    [1,3,2],
    [2,3,0],
    [2,0,0],
    [1,3,0]
    ])

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
operator_knowl = [[2,2,2], 
                  [3,1,1],
                  [1,2,3]]

operator_pref = [[3,1,3],
                 [2,3,2],
                 [1,3,1]]

#machining_data: matrix M x E with M number of machines, E levels of experience; entries are the average time spent (per unit) per knowledge level (cols) 
machining_data = cpm_array([
    [8,15,20],
    [6,8,10],
    [2,6,10]
    ])

"""
#unit_data: vector M --> entries are the unit (number of parts) for which the average execution times are given in machining_data
# REDUNDANT if unit == 1
unit_data = cpm_array([
    1,
    1,
    1])
"""

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

#tasks2machines: a vector with length T i.e. the number of tasks to be executed in total.
# the task ID is the corresponding index in the vector
# the entry is the machine ID for the task
machine2task = [] #vector, contains required machine for task with ID i
for i in range(0,len(parts_data)):
    this_part = sequence_data[i,:]
    nonzero = np.where(this_part > 0)
    this_part = this_part[nonzero]
    for j in range(parts_data[i]):
        machine2task = np.append(machine2task,this_part)
#number tasks that need to be executed
Ntasks = sum(tasks2machines)
#indeces of sorted machines
sortedIdx_m2t = np.argsort(machine2task)

  
"""  
#how many parts should be processed on each machine?
parts_per_machine = cpm_array(np.zeros([1,machines_count]))
for m in range(1,machines_count+1):
    #find indeces of jobs where machine is needed
    jidx = np.where(sequence_data == m)[0]
    #count the parts
    if len(jidx) > 0:
       parts_per_machine[0,m-1] = sum(parts_data[jidx]) 
"""
    
#whatis the upper bound ub of the make span? Therefore: take max make time (lowest knowledge level) times number of parts
ub_makespan = parts_per_machine * machine_data[:,-1]

#introduce a scaling factor for the ub of the makespan.
#why? Actual problem has a max makespan of 10 days; this toy example does not have a max makespan
#the scaling factor allows to study the influence of the max makespan on the solution
ub_scaler = 0.75    
max_makespan = int(ub_scaler * ub_makespan)

# How many parts should be produced by the same operator on the same machine (minimum)
lb_partspan = 1  

"""----------------------------------------------------------------------------
##-decision variables
"""
#here: keep track of each part
#we need a start-time for each part
#start times can overlap, as they link to different machines
start_time_task = intvar(0, max_makespan, shape=(1,sum_of_tasks))
end_time_task = intvar(0, max_makespan, shape=(1,sum_of_tasks))

#flow_parts2machines to know which machine is used to exe a task
flow_tasks2machines = machine2task #intvar(1, range_machines, shape=(1,sum_of_tasks))
#flow_parts2operators to know which operator is working on a task
flow_operators2tasks = intvar(1, range_operators, shape=(1,sum_of_tasks))
#derive which operator is working on which machine
#expertise level and preference level can be derived once operator ID is known

"""----------------------------------------------------------------------------
##-tuning parameters
"""
ub_ergo = 3
lb_allocation4operator = .7
weight_preference = 1
weight_expertlevel = 1
"""----------------------------------------------------------------------------
##-modelling
"""
model = Model()

###Constraints:

###START AND END CONDITION    
#condition on start time
model += (min(start_time_task) == 0)
#condition on end time     ===> is this usefull???
model += (max(end_time_task) <= max_makespan)

###SEQUENCE OF TASKS
## 1/ task allocation to machines should be correct.
# this is INPUT

## 2/ respect the sequence of the tasks (on the level of a part)
#if the first task is part of a sequence, then the second task can only start after the first task
model += [((machine2task[i] != machine2task[i+1] for i in range(1)) & (parts_data[0] > 1)).implies \
              (start_time_task[i+1] > end_time_task[i])]
#if the last task is part of a sequence, then the second task can only start after the first task
model += [((machine2task[i] != machine2task[i-1] for i in range(Ntasks-2,Ntasks)) & (parts_data[-1] > 1)).implies \
              (start_time_task[i] > end_time_task[i-1])]             
#if a task is part of a sequence, it cannot start before the previous task of the sequence and the next task of the sequence cannot
#start before the current task has finished: 
#MID of SEQUENCE
model += [(((machine2task[i] != machine2task[i+1]) & (machine2task[i] != machine2task[i-1]) for i in range(1,Ntasks-1))).implies \
            ((start_time_task[i+1] > end_time_task[i]) & (start_time_task[i] > end_time_task[i-1]))]
#START of SEQUENCE    
model += [(((machine2task[i] != machine2task[i+1]) for i in range(0,Ntasks-1))).implies \
            ((start_time_task[i+1] > end_time_task[i]))]
#END of SEQUENCE
model += [(((machine2task[i] != machine2task[i-1]) for i in range(1,Ntasks))).implies \
            ((start_time_task[i] > end_time_task[i-1]))]


###ALLOCATION OF OPERATORS
#an operator cannot be allocated to +1 task simultanuously
model += [(flow_operators2tasks[i] == flow_operators2tasks[j]) for i in range(0,Ntasks) for j in range(0,Ntasks).implies \
          ( (start_time_task[i] > end_time_task[j]) | (start_time_task[j] > end_time_task[i]))]

###DURATION OF A TASK
##production of part >> start of task + duration of task = end of task
#derive the execution time 
this_knowl = operator_knowl[flow_operators2tasks[i]][flow_tasks2machines[i]]
this_exetime = machining_data[flow_operators2tasks[i],this_knowl]
#The estimated end time of a task equals the start time of the task and the estimated execution time of the task
model += [(start_time_task[i] + this_exetime == end_time_part[i] for i in range(sum_of_tasks))]

### MIMIC ERGO CONSTRAINT
## 1 operator is allowed to work for 3 consequetive tasks on the same machine
#the operator working on a machine for task i, can be the same operator working on  task i+ub_ergo
#but then either the operator is not working on the same machine
#or the operator has in between between working on at least 1 different machine.
"""ub_ergo = 3"""
#derive the execution time 
this_knowl = operator_knowl[flow_operators2tasks[i]][flow_tasks2machines[i]]
this_exetime = machining_data[flow_operators2tasks[i],this_knowl]
model += [flow_operators2tasks[i] for i in range(0,Ntasks).implies \
          ((end_time_task[j] != (ub_ergo * this_exetime) + starttime[k]) & \
           (flow_operators2tasks[j] == flow_operators2tasks[k]) & \
           (flow_operators2tasks[k] == flow_operators2tasks[i])) for j in range(0,Ntasks) for k in range(0,Ntasks)    
        ]


### MAKESPANE < DESIRED MAX MAKESPAN
makespan = max(end_time_machine)
model += (makespan <= max_makespan)

### AN OPERATOR SHOULD BE AT LEAST ALLOCATED ON A MINIMUM OF TASKS
## here modelled as 0.7 * (no of tasks/no of operators)
"""lb_allocation4operator = .7"""
for i in range_operators:
    model += [np.count_nonzero(flow_operators2tasks == i+1) >= (lb_allocation4operator * Ntasks/operators_count)]

### objective
"""weight_preference = 1
weight_expertlevel = 1"""
##objective is to maximize the knowledge level and the preference level
# therefore: link the levels to the operator+machine combination per task
this_knowl = operator_knowl[flow_operators2tasks,flow_tasks2machines]
this_pref = operator_pref [flow_operators2tasks,flow_tasks2machines]
goal = weight_preference * np.sum(this_knowl) + weight_expertlevel * np.sum(this_pref) 
#min goal == -max goal
model.minimize(-goal)

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