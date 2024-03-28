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
operator_data = cpm_array([
    [[2,3],[2,1],[2,3]], #operator 1: for machine 1 --> knowledge level 2, preference level 3; for machine 2 --> knw 2, pref 1 , ...
    [[3,2],[1,3],[1,2]], #operator 2: ...
    [[1,1],[2,3],[3,1]], #operator 3: ...
    ])

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
tasks2machines = []
flat_parts_data = np.reshape(np.repeat(parts_data,max_sequence),(len(parts_data),max_sequence)).flatten()
flat_sequence_date = sequence_data.flatten()
nonzero = np.where(flat_sequence_date > 0)
flat_sequence_date = flat_sequence_date[nonzero]
flat_parts_data = flat_parts_data[nonzero]
for i in range(0,len(flat_parts_data)):
    tasks2machines = np.append(tasks2machines,flat_sequence_date[i]*np.ones([1,flat_parts_data[i]]))
#how much tasks should be executed all together?
sum_of_tasks = sum(tasks2machines)
  
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
# >> goto flow_parts2machines to know which machine is used
# >> goto flow_parts2operators to know which operator is working
#we need a start-time for each part
#start times can overlap, as they link to different machines
start_time_part = intvar(0, max_makespan, shape=(1,sum_of_tasks))
end_time_part = intvar(0, max_makespan, shape=(1,sum_of_tasks))

flow_parts2machines = intvar(1, sum_of_jobs, shape=(1,sum_of_tasks))
#expertise level and preference level can be derived once operator ID is known
flow_parts2operators = intvar(1, sum_of_jobs, shape=(1,sum_of_tasks))


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

###production of part >> start of task + duration of task = end of task
##derive the execution time
#first: get knowledge level for each tasks given the current combination (operators and machines)
this_knowl = operator_data[:,:,0][flow_parts2operators,flow_parts2machines]
this_pref = operator_data[:,:,1][flow_parts2operators,flow_parts2machines]
#second: get execution time of the task for the current combination (machines and knowledge level)
this_exetime = machining_data[flow_parts2machines,this_knowl]
#add the constraint
model += (start_time_part + this_exetime == end_time_part)

###operator can only be assigned to 1 task (no overlap in task allocation)
##a task can only be started by 1 operator and 1 machine
idx_sorted_operators = np.argsort(flow_parts2operators)
sorted_start_times = start_time_part[idx_sorted_operators]
sorted_end_times = sorted_end_times[idx_sorted_operators]
#overlap in starttime and endtime within an operator is forbidden! 
for o in (1,operators_count+1):
    idx_of_o = np.where(flow_parts2operators == o)
    o_startimes = np.sort(start_time_part[idx_of_o])
    o_endtimes = np.sort(end_time_part[idx_of_o])
    model += (o_endtimes[:-1] - o_startimes[1:])
    
###machine can only be assigned to 1 task (no overlap in machine allocation)
##a task can only be started by 1 operator and 1 machine
idx_sorted_machines = np.argsort(flow_parts2machines)
sorted_start_times = start_time_part[idx_sorted_machines]
sorted_end_times = sorted_end_times[idx_sorted_machines]
#overlap in starttime and endtime within a machine is forbidden! 
for m in (1,machines_count+1):
    idx_of_m = np.where(flow_parts2machines == m)
    m_startimes = np.sort(start_time_part[idx_of_m])
    m_endtimes = np.sort(end_time_part[idx_of_m])
    model += (m_endtimes[:-1] - m_startimes[1:])

###sequence of the tasks must be respected
##therefore: check per job whether a sequence constraint is relevant
#add sequence constraint
from_task = 0
to_task = 0
for j in range(0,jobs_count):
    job_sequence = jobs_data[j:]
    job_sequence = job_sequence[job_sequence != 0]
    #keep track of tasks
    to_task = from_task + parts_data[j] * len(job_sequence) 
    this_start_time = start_time_part[from_task : to_task]
    this_end_time = end_time_part[from_task : to_task]
    this_machineflow = flow_parts2machines[from_task : to_task]
    order = np.repeat(job_sequence, parts_data[j])
    this_order = np.transpose(order.reshape(job_sequence,int(len(order)/job_sequence))).flatten()
    #stort start times of the task from the selected job (increasing start time)
    idx_sorted_starttimes = np.argsort(this_start_time)
    #use the indeces to sort the machine per tasks
    sorted_machineflow = this_machineflow(idx_sorted_starttimes)
    #impose the order of the starttimes i.e. the sequence >> hard constraint
    model += (sorted_machineflow == this_order)
    from_task = to_task

### mimic an ergonomic constraint
## therefore: restrict number of repetitions, here: restriction on executing the same tasks more than 3x in row by 1 single operator
"""ub_ergo = 3"""
idx_order_machines = np.argsort(flow_parts2machines)
order_operators = flow_parts2operators[idx_order_machines]
#compare each ergo upper bound e.g. each ith element to each (i+1)th element: 
# if they are the same, i.e. di >> ergo constraint is broken
#is_ergo_satified = order_operator[ub_ergo :] - order_operators[(ub_ergo-1): -1]
model += (order_operator[ub_ergo :] != order_operators[(ub_ergo-1): -1])


###constrain the makespan as makespan <= max_makespan
makespan = max(end_time_machine)
model += (makespan <= max_makespan)

### operators should have a minimum allocation each
## here modelled as 0.7 * (no of tasks/no of operators)
"""lb_allocation4operator = .7"""
equal_tasks2operators = sum_of_tasks/operators_count
relax_tasks2operators = lb_allocation4operator * equal_tasks2operators
for o in range(1,operators_count+1):
    model += (len(np.where(flow_parts2operators == o)) >= relax_tasks2operators) 

### objective
"""weight_preference = 1
weight_expertlevel = 1"""
##objective is to maximize the knowledge level and the preference level
# therefore: link the levels to the operator+machine combination per task
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