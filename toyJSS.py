#!/usr/bin/python3
import numpy as np
from itertools import combinations, permutations
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
jobs_data = cpm_array([#sequence is max 3
    [5,2,0,0], # number of parts, sequence of machines
    [10,1,3,2],
    [15,2,3,0],
    [10,2,0,0],
    [5,1,3,0]
    ])

operator_data = cpm_array([#dim operator x machines x level type = 3 x 3 x 2
    [[2,3],[2,1],[2,3]], #operator 1
    [[3,2],[1,3],[1,2]], #operator 2
    [[1,1],[2,3],[3,1]], #operator 3
    ])

machine_data = cpm_array([#time spent for 1 part cols 1 per knowledge level cols 1,2,3
    [1,8,15,20],
    [1,6,8,10],
    [1,2,6,10]
    ])
###-

all_jobs_count = sum(jobs_data[:,0])
jobs_count = jobs_data.shape[0]
machines_count = machine_data.shape[0]
operators_count = operator_data.shape[0]

parts_on_machine_count = cpm_array(np.zeros([1,machines_count]))
for j in range(0,jobs_count,1):
    this_job = jobs_data[j,1:machines_count+1]
    this_job_parts = jobs_data[j,0]
    for m in range(0,machines_count,1):
        parts_on_machine_count[0,m] += this_job_parts * this_job[m]

ub_makespan_part = machine_data[:,3]
ub_makespan = 0
for m in range(0,machines_count,1):
    ub_makespan += ub_makespan_part[m] *  parts_on_machine_count[0,m]

ub_scaler = 0.75    
max_makespan = int(ub_scaler * ub_makespan)

##-decision variables
start_time_machine = intvar(0, max_makespan, shape=(machines_count,all_jobs_count))
end_time_machine = intvar(0, max_makespan, shape=(machines_count,all_jobs_count))

start_time_operator = intvar(0, max_makespan, shape=(operators_count,all_jobs_count))
end_time_operator = intvar(0, max_makespan, shape=(operators_count,all_jobs_count))

makespan_preference = intvar(1, 3, shape=(operators_count,max_makespan))
makespan_expertlevel = intvar(1, 3, shape=(operators_count,max_makespan))
makespan_allocation = intvar(0, 1, shape=(operators_count,max_makespan))
makespan_machine_allocation = intvar(0, 1, shape=(machines_count,max_makespan))

flow_parts2machines = intvar(1, 3, shape=(all_jobs_count,max_makespan))
flow_parts2operators = intvar(1, 3, shape=(all_jobs_count,max_makespan))

part_span = 1  # no consecutive parts made on same machine by same operator

all_jobs = range(all_jobs_count)
all_machines = range(machines_count)
all_operators = range(operators_count)

##-modelling
model = Model()

# start + dur = end
model += (start_time_machine + machine_data[:,1:machine_data.shape[1]-1] == end_time_machine)
model += (start_time_operator + machine_data[:,1:machine_data.shape[1]-1] == end_time_operator)
model += (start_time_machine == start_time_operator)
model += (end_time_machine == end_time_operator)

# Sequence constraints per job
for j in range(0,jobs_data.shape[0]):
    this_job = np.array(jobs_data[j,1:jobs_data.shape[1]]) #machine sequence for the job
    this_job = this_job[this_job != 0]
    if j == 0: this_job_parts = [j,jobs_data[j,0]]
    else: this_job_parts = [np.sum(jobs_data[0:j-1,0]),jobs_data[j]]
        #constraints on number of parts
        #sequence constraints
    for p in range(1,machines_count+1):
        perm_machines = permutations(np.arange(1,1+machines_count),p)
        for this_perm in list(perm_machines):
            #does permutation matches machine sequence?
            if this_perm == this_job:
                for s in range(0,all_jobs_count):
                    this_sequence = flow_parts2machines[s,np.nonzero(flow_parts2machines[s,:])]
                    # if this_job.shape[0] == 1:
                    #     model += (this_sequence)
                    for i in range(1,this_sequence.shape[0]-1):
                        # constraints on the sequence of the machines given the job
                        if this_job.shape[0] == 1:
                            model += (flow_parts2machines[s,i] == flow_parts2machines[s,i+end_time_machine]) | \
                                     (flow_parts2machines[s,i] == flow_parts2machines[s,i-end_time_machine]) 
                        elif this_job.shape[0] == 2:  
                            if flow_parts2machines[s,i] == this_job[0]:
                                model += (this_job[1] == flow_parts2machines[s,i+end_time_machine]) | \
                                         (this_job[1] == flow_parts2machines[s,i-end_time_machine]) 
                            if flow_parts2machines[s,i] == this_job[1]:
                                model += (this_job[0] == flow_parts2machines[s,i+end_time_machine]) | \
                                         (this_job[0] == flow_parts2machines[s,i-end_time_machine]) 
                        elif this_job.shape[0] == 3:  
                            if flow_parts2machines[s,i] == this_job[0]:
                                model += (this_job[1] == flow_parts2machines[s,i+end_time_machine]) | \
                                         (this_job[2] == flow_parts2machines[s,i-end_time_machine]) 
                            if flow_parts2machines[s,i] == this_job[1]:
                                model += (this_job[2] == flow_parts2machines[s,i+end_time_machine]) | \
                                         (this_job[0] == flow_parts2machines[s,i-end_time_machine]) 
                            if flow_parts2machines[s,i] == this_job[2]:
                                model += (this_job[0] == flow_parts2machines[s,i+end_time_machine]) | \
                                         (this_job[1] == flow_parts2machines[s,i-end_time_machine]) 

# # Precedence constraint per job
# for m1,m2 in combinations(all_machines,2): # [(0,1), (0,2), (1,2)]
#     print(m1)
#     print(m2)
#     model += (end_time[m1,:] <= start_time[m2,:])

# # No overlap constraint: one starts before other one ends
# for j1,j2 in combinations(all_jobs, 2):
#     model += (start_time[:,j1] >= end_time[:,j2]) | \
#               (start_time[:,j2] >= end_time[:,j1])


# makespan <= max_makespan
makespan = max(end_time_machine)
model += (makespan <= max_makespan)

# allocation of operators > 70% of total make time
model += (np.sum(makespan_allocation,axis=1)/makespan >= 0.7) 

# objective
weight_preference = 1
weight_expertlevel = 1
goal = -sum(weight_preference * np.sum(makespan_preference,axis=1) + weight_expertlevel * np.sum(makespan_expertlevel,axis=1)) 

model.minimize(goal)

#print(model.status())
val = model.solve()
print("Makespan:",makespan.value())
print("Schedule:")
grid = -8*np.ones((machines_count, makespan.value()), dtype=int)
for j in all_jobs:
    for m in all_machines:
        grid[m,start_time[m,j].value():end_time[m,j].value()] = j
print(grid)
