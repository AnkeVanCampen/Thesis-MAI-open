# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 16:27:24 2024

@author: anke
"""
import cpmpy as cp
import numpy as np
from itertools import combinations, permutations, repeat,combinations_with_replacement
from cpmpy import *
# import gurobipy as gp
# print(gp.gurobi.version())
import matplotlib.pyplot as plt
from itertools import zip_longest

#VISUALIZATION via GANTT: small errors to be fixed in calculating end times of shifts
def next_multiple_of_4(x):
    return x + (4 - x % 4) if x % 4 != 0 else x + 4
def previous_multiple_of_4(x):
    return x - (x % 4)
def draw_operators(operator,start_here,end_here,ax,Ocolors,machine, num_operators):

            ax.barh(y=machine - 0.1, left=int(start_here), width=int(end_here - start_here), height=0.2, 
                    color=Ocolors(operator % num_operators), edgecolor="black")
            ax.text(int(start_here + (end_here - start_here) / 2), machine -.1, f"O{operator}",
                    va='center', ha='center', color="black", fontsize=8, fontweight='bold')
def plot_shift_schedule(start_times, end_times, machine_assignments, val_operator2halfshift, val_tasks2halfshift, num_operations, num_machines, num_operators, shift_duration, num_shifts):
    fig, ax = plt.subplots(figsize=(14, 7))

    major_ticks = np.arange(0, (num_shifts + 1) * shift_duration, shift_duration)  # Full shifts
    minor_ticks = np.arange(shift_duration / 2, num_shifts * shift_duration, shift_duration)  # Half shifts

    Tcolors = plt.colormaps.get_cmap("tab10")  # Different colors for each operations
    Ocolors = plt.colormaps.get_cmap("tab20")  # Different colors for each operator

    for i in range(num_operations):
        start = start_times[i]
        end = end_times[i]
        machine = machine_assignments[i]
        
        for s in range(num_shifts):
            for h in range(2):
                task_in_halfshift = val_tasks2halfshift[i, s, h]
                operator_in_halfshift = val_operator2halfshift[i, s, h]
                if not task_in_halfshift:
                    continue  
                halfshift_start = s * shift_duration + (h * shift_duration / 2)
                halfshift_end = halfshift_start + (shift_duration / 2)
                overlap_time = max(0, min(halfshift_end, end) - max(start, halfshift_start))
                if overlap_time > 0:
                    if start < halfshift_start:
                        op_start = halfshift_start
                    elif start > halfshift_start:
                        op_start = start
                    elif start == halfshift_start:
                        op_start = start
                    if end > halfshift_end:
                        op_end = op_start + shift_duration / 2
                    elif end < halfshift_end:
                        op_end = end
                    elif end == halfshift_end:    
                        op_end = end

                    draw_operators(operator_in_halfshift, op_start, op_end, ax, Ocolors, machine, num_operators)

        ax.barh(y=machine+.1, left=start, width=end - start, height=0.2, 
               color=Tcolors(i % num_operations), edgecolor="black")   
        ax.text(start + (end - start) / 2, machine+.1, f"T{i}",
                va='center', ha='center', color="white", fontsize=8, fontweight='bold')


    for s in range(1, num_shifts):
        plt.axvline(x=s * shift_duration, color="gray", linestyle="--", alpha=0.8)
        plt.axvline(x=(s - 1) * shift_duration + shift_duration / 2, color="red", linestyle=":", alpha=0.4)

    ax.set_yticks(np.arange(num_machines))
    ax.set_yticklabels([f"Machine {m+1}" for m in range(num_machines)])

    ax.set_xticks(major_ticks)  
    ax.set_xticks(minor_ticks, minor=True)  
    
    ax.set_xticklabels([f"{8*i}" for i in range(len(major_ticks))])
    ax.set_xticklabels([f"{8*i-4}" for i in range(1, len(minor_ticks) + 1)], minor=True)

    ax.tick_params(axis='x', which='major', length=10, width=2, labelsize=10)  
    ax.tick_params(axis='x', which='minor', length=5, width=1, labelsize=8) 

    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Machines")
    ax.set_title("Shift Schedule - Operators, Machines & Tasks")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

#DATA: dummy data as input
#INPUT: generated according paper for benchmarking'
#from Ghorbani Saber et al. 2023, 2023, MIP for integrated FJSS in flex manufact systems'
#5 machines, 6 operators, 12 shifts of 8h'
sequence_data = cpm_array([
    [1,2,3,4,5], 
    [6,7,8,9,10],
    [11,12,13,14,15],
    [16,17,18,19,20],
    [21,22,23,24,25]
    ])

#rows = operators, cols = machines
operator_knowl = cpm_array([
    [1,-1,1,-1,-1], 
    [-1,1,1,-1,1],
    [-1,-1,-1,1,1],
    [1,1,-1,1,-1],
    [1,-1,1,1,-1],
    [-1,1,-1,-1,1]
    ])

#processing time on the machine for the operation, in hours
machining_data = cpm_array([
    [-1,-1,1,-1,-1], #originally 1 instead of 10
    [3,-1,-1,-1,-1],
    [-1,6,-1,-1,-1],
    [-1,-1,-1,7,-1],
    [-1,-1,-1,3,3],
    [-1,8,-1,-1,-1],
    [-1,-1,5,-1,-1],
    [-1,-1,-1,-1,10],
    [-1,-1,-1,10,-1],
    [10,-1,-1,-1,-1],
    [-1,-1,5,-1,5],
    [-1,-1,-1,4,-1],
    [-1,8,-1,-1,-1],
    [9,-1,-1,-1,-1],
    [-1,1,-1,-1,1],
    [-1,-1,9,-1,-1],
    [-1,3,-1,-1,-1],
    [-1,-1,-1,-1,5],
    [-1,4,-1,4,-1],
    [3,-1,-1,-1,-1],
    [-1,3,-1,3,-1],
    [-1,-1,-1,3,-1],
    [-1,-1,9,-1,-1],
    [10,-1,-1,-1,-1],
    [-1,-1,-1,-1,4]
    ])


#Derive some parameters
#Get an idea of max makespan
array = np.array(machining_data)
mask = array[array != -1]
min_time = min(mask)
max_time = max(mask)

#Count and set ranges: operatorsm machines, jobs, tasks, shift
num_operators = np.shape(operator_knowl)[0]
range_operators = range(num_operators)

num_machines = np.shape(machining_data)[0]
range_workcenters = range(num_machines)

num_jobs = np.shape(sequence_data)[0]
num_tasks = np.shape(sequence_data)[1]

num_operations = num_jobs*num_tasks
range_operations = range(num_operations)

#INPUT PARAMETERS
shift_duration = 8 #hours per block
num_shifts = 20
range_shifts = range(num_shifts)
ub_ergo = 4 #at most 4 consecutive hours on a machine
lb_allocation_full = 5 #at least 5h allocation within a full shift > otherwise penalize
lb_allocation_half = 3 #at least 5h allocation within a half shift > otherwise penalize
desired_makespan = 8*12 #which makespan do you target?
#weights for the soft constraints
pen_weight_shifts = 100 #switching shifts
pen_weight_switch = 10 #switching machines
pen_weight_ergo = 1 #ergo
pen_weight_allocation = 1 #fairness allocation
pen_weight_makespan = 10000 #makespan importance

##DECISION VARIABLES
start_time = intvar(0, int(ub_makespan), shape=num_operations, name="start")
end_time = intvar(0, int(ub_makespan), shape=num_operations, name="stop")
execution_time = intvar(0,100,shape=num_operations)

flow_machine2job = intvar(0, num_machines - 1, shape=num_operations, name="machine_to_job")
flow_tasks2shift = boolvar(shape=(num_operations,num_shifts), name="task_to_shift")      
flow_tasks2halfshift = boolvar(shape=(num_operations,num_shifts,2), name="task_to_halfshift")
flow_operator2halfshift = intvar(-1, num_operators-1, shape=(num_operations,num_shifts,2), name="operator_to_halfshift")   
#flow_knowledge2halfshift = intvar(-1, 4, shape=(num_operations,num_shifts,2), name="knowledge_to_halfshift")   
  
#SOFT CONSTRAINTS
penalty_vars_shifts = intvar(0, 1000, shape=(num_operators, num_shifts - 2), name="shift_penalty")
penalty_vars_switch = intvar(0, 1000, shape=(num_operators, num_shifts), name="machine_switch_penalty")
penalty_vars_ergo = intvar(0, 1000, shape=(num_operators, num_shifts, num_machines), name="ergo_penalty")
penalty_vars_allocation = intvar(0, 1000, shape=(num_operators, num_shifts), name="operator_alocation_penalty")
total_operator_time = intvar(0, block_duration, shape=(num_operators, num_shifts,2))
penalty_vars_makepan = intvar(0, int(ub_makespan), name="makespan_penalty")


weight_makespan = pen_weight_makespan/desired_makespan


###MODEL
from enum import Enum
from functools import total_ordering
import time
model = Model()

###Constraints:

#CONSTRAINT the first task should start at timestep == 0
model += (min(start_time) == 0)

           
# CONSTRAINT
# the end time of a task equals the start time + execution time
# exetime depends on (operator, machine and knowledge level of the operator for the machine)
# Set execution time based on operator and machine combinations
for i in range_operations:
    model += sum(#sum works as logical or --> At least one valid (machine, operator) combination
        (flow_machine2job[i] == m) & 
        (operator_knowl[o, m] == 1) & 
        (machining_data[i, m] > -1)
        for m in range_workcenters for o in range_operators
    ) >= 1

    #Ensure correct execution time given operator and machine combination
    for m in range_workcenters:
        #if an operator works with machine m on job i, then the execution time for i is given by the machining data
        #REMARK: the constraint is unidirectional ==> , and hence not <==> 
        model += (flow_machine2job[i] == m).implies(execution_time[i] == machining_data[i, m])
    
#CONSTRAINT
for i in range_operations:
    model += start_time[i] + execution_time[i] == end_time[i]

#check pairs of operations: complexity is O(n2)  
for i in range(num_operations):
    for j in range(i + 1, num_operations):  # Compare each pair of operations

        for s in range_shifts:

            for h in range(2):  # 0 = first half, 1 = second half
                model += (
                    (flow_operator2halfshift[i, s, h] == flow_operator2halfshift[j, s, h])  
                    & (start_time[i] < end_time[j])  
                    & (start_time[j] < end_time[i])  
                    & (flow_tasks2halfshift[i, s, h] == 1)  
                    & (flow_tasks2halfshift[j, s, h] == 1)  
                    ).implies(False)
                # Machine conflict: same machine, overlapping time
                model += (
                    (flow_machine2job[i] == flow_machine2job[j])  #check if tasks are on the same machine
                    & (start_time[i] < end_time[j]) #these 2 constraints ensure that the task do overlap in time 
                    & (start_time[j] < end_time[i])  
                    & (flow_tasks2halfshift[i, s, h] == 1) #both tasks are running in the same half shift 
                    & (flow_tasks2halfshift[j, s, h] == 1)  
                ).implies(False)
 
#an  operator cannot continue working at task over the shift boundary
for i in range(num_operations):  
    for s in range(num_shifts):
        for h in range(2):
            #start and end time of the second half shift
            halfshift_start = s * shift_duration + h * int((shift_duration / 2))
            halfshift_end = halfshift_start + int((shift_duration / 2))
            
            #a task is assigned to the half shift if the task is active in the half shift (overlap)
            model += (flow_tasks2halfshift[i, s, h] == 1).implies(
                (start_time[i] < halfshift_end) & (end_time[i] > halfshift_start)
                )
            model += (flow_operator2halfshift[i, s, h] != -1).implies(
                (flow_tasks2halfshift[i, s, h] == 1)
                )

           #a if no overlap, than the task is not assigned to the half shift
            model += (flow_tasks2halfshift[i, s, h] == 0).implies(
                (start_time[i] >= halfshift_end) | (end_time[i] <= halfshift_start)
                )    
            model += (flow_operator2halfshift[i, s, h] == -1).implies(
                (flow_tasks2halfshift[i, s, h] == 0)
                )
            # model += (flow_knowledge2halfshift[i, s, h] == -1).implies(
            #     (flow_tasks2halfshift[i, s, h] == 0)
            #     )

            if h == 1 and s < (num_shifts-1): 
                if end_time[i] > halfshift_end: #this means the task continues in the next shift
                    model += (flow_tasks2halfshift[i, s, 1] == 1).implies(flow_operator2halfshift[i,s,1] != flow_operator2halfshift[i,s + 1, 0]) 
                if end_time[i] > halfshift_end + shift_duration/2:   
                    model += (flow_tasks2halfshift[i, s, 1] == 1).implies(flow_operator2halfshift[i,s,1] != flow_operator2halfshift[i,s + 1, 1]) 
            

#CONSTRAINT: respect the sequence of tasks for a certain jobs
for j in range(num_jobs):
    job_tasks = sequence_data[j]  # this is the order for current job

    for t in range(1, 5):  # loop over the subtasks starting from the second subtask
        prev_task = job_tasks[t - 1]
        current_task = job_tasks[t]
        
        # if -1 in the sequence: no task left in the sequence >> continue to next job
        if current_task == -1 or prev_task == -1:
            continue
        
        # current task should start after previous task
        model += start_time[current_task-1] >= end_time[prev_task-1]

# # #HARD CONSTRAINT: consecutive shifts: if operator works in shift 0, he cannot work in shifts -2,-1,1,2 
# for o in range_operators:
#     for s in range(num_shifts - 2): 
#             #stay in the bounds: if s = 0, than s-1 and s-2 are not valid
#             valid_shifts = [s + delta for delta in [-2, -1, 1, 2] if 0 <= s + delta < num_shifts]
#             #if an operator is assigned to shift s, than the only valid shift he can work is, is shift s
#             model += (sum(flow_operator2shift[:, s] == o) > 0).implies(
#                        sum(flow_operator2shift[:, valid_shifts] == o) == 0 )

#SOFT CONSTRAINT: consecutive shifts: if operator works in shift 0, he cannot work in shifts -2,-1,1,2 
for o in range_operators:
    for s in range(num_shifts - 2): 

            #put it relative to first half shift
            #stay in the bounds: if h = 0, than s-1 for h = 0 and 1 and s-2 for h = 0 and 1 are not valid
            shifts2avoid_plus1 = [s + delta for delta in [-1, 1] if 0 <= s + delta < num_shifts]
            shifts2avoid_plus2 = [s + delta for delta in [-2, 2] if 0 <= s + delta < num_shifts]
            #count in how many forbidden shifts the operator is allocated
            shift_violations_plus1 = sum(flow_operator2halfshift[:, f, :] == o for f in shifts2avoid_plus1)
            shift_violations_plus2 = sum(flow_operator2halfshift[:, f, :] == o for f in shifts2avoid_plus2)
            #if an operator works in s, then penalyze this if he also works in the shifts2avoid
            model += (sum(flow_operator2halfshift[:, s, :] == o) > 0).implies(penalty_vars_shifts[o, s] == pen_weight_shifts * shift_violations_plus1)
            model += (sum(flow_operator2halfshift[:, s, :] == o) > 0).implies(penalty_vars_shifts[o, s] == int(pen_weight_shifts/2) * shift_violations_plus2)
   
           
# #SOFT CONSTRAINT: reduce number of switches between machines within a shift for an operator
# """
# for o in range_operators:
#     for s in range(num_shifts):  
        
#         #which machines are used?
#         for m in range(num_machines):
#             #model += (machine_used[m] == any(flow_machine2job[i] == m for i in range(num_operations) if flow_operator2shift[i, s] == o))
#             for h in range(2):
#                 model += (machine_used[m] == any(flow_machine2job[i] == m for i in range(num_operations) if flow_operator2halfshift[i, s, h] == o))

#         #how many uniques?
#         model += (num_unique_machines == sum(machine_used))
        
#         # put a penalty on the machine switches (number of unique machines - 1)
#         model += (num_unique_machines-1 == 0).implies(penalty_vars_switch[o, s] == 0)
#         model += (num_unique_machines-1 == 1).implies(penalty_vars_switch[o, s] == int(pen_weight_switch/2))
#         model += (num_unique_machines-1 > 1).implies(penalty_vars_switch[o, s] == int(pen_weight_switch))
# """        
# for o in range_operators:
#     for s in range(num_shifts):
#         #make a list of machines, initialized to 0 = not used
        
#         for i in range(num_operations):
#             for h in range(2):  
#                 if flow_operator2halfshift[i, s, h] == o:
#                     model += (machine_used[flow_machine2job[i]] == 1)
#                 else:    
#                     model += (machine_used[flow_machine2job[i]] == 0)
#         #for the operator in the shift, count the number of machines
#         num_unique_machines = sum(machine_used)  
        
#         # set the penalty on the switches
#         model += penalty_vars_switch[o, s] == (max(0, num_unique_machines - 1) * pen_weight_switch)
            

# #SOFT CONSTRAINT: avoid working for longer than 4h on the same machine
for s in range(num_shifts):
    start_shift = shift_duration * s               
    end_shift = start_shift + shift_duration

    for i in range_operations:
        ergo_overlap_time = max(0, min(end_shift, end_time[i]) - max(start_shift, start_time[i]))

        penalty_value = ((ergo_overlap_time > 4) & (flow_operator2halfshift[i, s, 0] == flow_operator2halfshift[i, s, 1])) * (ergo_overlap_time - ub_ergo) * pen_weight_ergo

        for o in range_operators:
            model += penalty_vars_ergo[o, s, flow_machine2job[i]] == penalty_value

      
#SOFT: allocation fairness
for o in range_operators:
    for s in range(num_shifts):
        for h in range(2):
            total_time = 0
            for i in range_operations:
                if flow_operator2halfshift[i, s, h] == o:
                    shift_start = s * shift_duration + (h * int(shift_duration / 2))
                    shift_end = shift_start + int(shift_duration / 2)
                    overlap_time = max(0, min(end_time[i], shift_end) - max(start_time[i], shift_start))
                    total_time += overlap_time
            model += (total_operator_time[o, s, h] == total_time)
for o in range_operators:
    for s in range(num_shifts):
        
        # check the half shifts, put pen
        model += ((total_operator_time[o, s, 0] > 0) & (total_operator_time[o, s, 1] == 0) & (total_operator_time[o, s, 0] < lb_allocation_half)).implies(
            penalty_vars_allocation[o, s] == pen_weight_allocation * (lb_allocation_half - total_operator_time[o, s, 0])
            )
        model += ((total_operator_time[o, s, 1] > 0) & (total_operator_time[o, s, 0] == 0) & (total_operator_time[o, s, 1] < lb_allocation_half)).implies(
            penalty_vars_allocation[o, s] == pen_weight_allocation * (lb_allocation_half - total_operator_time[o, s, 1])
            )
               
        # check the shift, put pen
        model += ((total_operator_time[o, s, 0] > 0) & (total_operator_time[o, s, 1] > 0) & (sum(total_operator_time[o, s, :] for h in range(2)) < lb_allocation_full)).implies(
            penalty_vars_allocation[o, s] == pen_weight_allocation * (lb_allocation_full - sum(total_operator_time[o, s, :] for h in range(2)))
        )



print('================SOLVING CP====================')
#FAST: but results in EMPTY shifts
goal_makespan = max(0,(max(end_time) - int(desired_makespan)))*pen_weight_makespan
# #VERY SLOW
#goal_makespan = max(end_time)*weight_makespan #
#goal_knowledge = sum(flow_knowledge2halfshift)
model.minimize(goal_makespan  
               + sum(penalty_vars_shifts)
               #+ sum(penalty_vars_switch) 
               + sum(penalty_vars_ergo) 
               + sum(penalty_vars_allocation)
               ) 


from cpmpy.solvers import CPM_ortools
print('--ORTOOLS--')
solver = CPM_ortools(model)  # OR-Tools solver
val = solver.solve()
print(val)
print(solver.status())

if model.solve():
    print("Solution found!")
    plot_shift_schedule(start_time.value(), end_time.value(), flow_machine2job.value(), flow_operator2halfshift.value(), flow_tasks2halfshift.value(), num_operations, num_machines, num_operators, shift_duration, num_shifts)
    goal_makespan_value = goal_makespan.value()
    penalty_shifts_value = sum(var.value() for var in penalty_vars_shifts)
    #penalty_switch_value = sum(var.value() for var in penalty_vars_switch)
    penalty_ergo_value = sum(var.value() for var in penalty_vars_ergo)
    penalty_allocation_value = sum(var.value() for var in penalty_vars_allocation)
    print('goal_makespan_value ',goal_makespan_value)
    print('penalty_shifts_value ',sum(penalty_shifts_value))
    #print('penalty_switch_value ',sum(penalty_switch_value))
    print('penalty_ergo_value ',sum(sum(penalty_ergo_value)))
    print('penalty_allocation_value ',sum(penalty_allocation_value))
else:
    print("No solution found.")


