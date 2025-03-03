# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 16:27:24 2024

!!!!! ADAPTIVE LEARNING APPROACH

@author: anke
"""
#restart kernel when running script
from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf') 

import os
os.environ['GRB_LICENSE_FILE'] = r'C:\Users\maarten\gurobi.lic'  # Replace with the correct path to your license file

import gurobipy
print(gurobipy.__file__)

import gurobipy as gp
print(gp.gurobi.version())

import gurobipy as gp
try:
    m = gp.Model("test")
    print("Gurobi license is working correctly!")
except gp.GurobiError as e:
    print(f"Error: {e}")



import cpmpy as cp
import numpy as np
from itertools import combinations, permutations, repeat,combinations_with_replacement
from cpmpy import *
# import gurobipy as gp
# print(gp.gurobi.version())
import matplotlib.pyplot as plt
from itertools import zip_longest

solver = SolverLookup.get("gurobi")
#solver = 'ortools'

'INPUT: generated according paper for benchmarking'
'from Ghorbani Saber et al. 2023, 2023, MIP for integrated FJSS in flex manufact systems'
'5 machines, 6 operators, 12 shifts of 8h'
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

num_operators = 6
range_operators = range(num_operators)
num_machines = 5
range_workcenters = range(num_machines)
num_jobs = np.shape(sequence_data)[0]
num_tasks = 5
num_operations = num_jobs*num_tasks
range_operations = range(num_operations)

shift_duration = 8
num_shifts = 14
desired_makespan = shift_duration*num_shifts
range_shifts = range(num_shifts)

ub_ergo = 4 #at most 4 consecutive hours on a machine
                                    
lb_allocation = 5 #at least 5h per shift allocated
lb_allocation_half = 3 #lb for halfshift

#PARAMETERS FOR SOFT CONSTRAINTS
pen_weight_shifts = 10
pen_weight_switch = 10
pen_weight_ergo = 10
pen_weight_allocation = 10
pen_weight_makespan = 1
pen_weight_knowl = 0


#VISUALIZATION via GANTT
def next_multiple_of_4(x):
    return x + (4 - x % 4) if x % 4 != 0 else x + 4

def previous_multiple_of_4(x):
    return x - (x % 4)

def detect_empty_shifts(val_tasks2half):
    shifts = np.sum(val_tasks2half,axis=2)
    full_shifts = np.sum(shifts,axis=0)
    non_zero_indices = np.where(full_shifts > 0)[0]
    print("non zero shifts: ",non_zero_indices)
    last_nonzero = non_zero_indices[-1]
    print("last nonzero shifts: ",last_nonzero)
    zero_indices = np.where(full_shifts==0)[0]
    print("zero shifts: ",zero_indices)
    empty_shifts = np.where(zero_indices < last_nonzero)[0]
    print("zero shifts: ",zero_indices)
    num_empty_shifts = len(empty_shifts)
    return num_empty_shifts
    
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

from enum import Enum
from functools import total_ordering
import time
main_obj = 0 #1,2,3,4,5
prev_solution = None
iteration = 0
num_iterations = 3
incl_knowledge = None
incl_ergo = None
incl_shifts = 1
incl_switch = None
incl_allocation = None
while iteration < num_iterations:
    
    
    # start_time = intvar(0, int(desired_makespan), shape=num_operations, name="start_time")
    # end_time = intvar(0, int(desired_makespan), shape=num_operations, name="end_time")

    # #flow_machine2task: wich machine for wich task 
    # flow_machine2job = intvar(0, num_machines - 1, shape=num_operations, name="flow_machine2job")

    # #flow_tasks2shift = boolvar(shape=(num_operations,num_shifts), name="flow_tasks2shift")      
    # flow_tasks2halfshift = boolvar(shape=(num_operations,num_shifts,2), name="flow_tasks2halfshift")
    # flow_operator2halfshift = intvar(-1, num_operators-1, shape=(num_operations,num_shifts,2), name="flow_operator2halfshift")   
    
    # execution_time = intvar(0,100,shape=num_operations,name="execution_time")
    initstart_time = time.time()
    print("INITIALIZATION OF DECISION VARIABLES")
    soft_warm_start_obj = 0
    if prev_solution is None:
        #print("Reinitializing start_time!")
        start_time = intvar(0, int(desired_makespan), shape=num_operations, name="start_time")
    else:    
        for idx, val in enumerate(prev_solution["start_time"]):
            soft_warm_start_obj += abs(start_time[idx] - val)
            #warm_start_constraints.append(start_time[idx] == val)
    if prev_solution is None:
        #print("Reinitializing end_time!")
        end_time = intvar(0, int(desired_makespan), shape=num_operations, name="end_time")
    else:            
        for idx, val in enumerate(prev_solution["end_time"]):
            soft_warm_start_obj += abs(end_time[idx] - val)
            #warm_start_constraints.append(end_time[idx] == val)
    if prev_solution is None:
        #print("Reinitializing flow_machine2job!")
        flow_machine2job = intvar(0, num_machines - 1, shape=num_operations, name="flow_machine2job")
    else:            
        for idx, val in enumerate(prev_solution["flow_machine2job"]):
            soft_warm_start_obj += (abs(flow_machine2job[idx] - val))
            #warm_start_constraints.append(flow_machine2job[idx] == val)
    if prev_solution is None:
        #print("Reinitializing flow_tasks2halfshift!")
        flow_tasks2halfshift = boolvar(shape=(num_operations,num_shifts,2), name="flow_tasks2halfshift")
    else:    
        for idx, val in enumerate(prev_solution["flow_tasks2halfshift"]):
            #soft_warm_start_obj += (abs(flow_tasks2halfshift[idx] - val))
            flow_tasks2halfshift = boolvar(shape=(num_operations,num_shifts,2), name="flow_tasks2halfshift")
            #warm_start_constraints.append(flow_tasks2halfshift[idx] == val)
    if prev_solution is None:
        #print("Reinitializing flow_operator2halfshift!")
        flow_operator2halfshift = intvar(-1, num_operators-1, shape=(num_operations,num_shifts,2), name="flow_operator2halfshift")   
    else:            
        for idx, val in enumerate(prev_solution["flow_operator2halfshift"]):
            soft_warm_start_obj += (abs(flow_operator2halfshift[idx] - val))
    if prev_solution is None:
        #print("initializing execution_time!")
        execution_time = intvar(0,100,shape=num_operations,name="execution_time")  
    else:    
        for idx, val in enumerate(prev_solution["execution_time"]):
            soft_warm_start_obj += (abs(execution_time[idx] - val))

    
    if iteration == 0:
        varmap = {"start_time":start_time, "end_time":end_time, "flow_machine2job":flow_machine2job, "flow_tasks2halfshift": flow_tasks2halfshift,
                  "flow_operator2halfshift": flow_operator2halfshift, "execution_time": execution_time}
    if iteration > 0:
        if incl_knowledge:
            if "flow_knowledge2halfshift" not in prev_solution:
                flow_knowledge2halfshift = intvar(0, 1, shape=(num_operations,num_shifts,2), name="flow_knowledge2halfshift")
                varmap["flow_knowledge2halfshift"] = flow_knowledge2halfshift
            else:    
                for idx, val in enumerate(prev_solution["flow_knowledge2halfshift"]):
                    soft_warm_start_obj += (abs(flow_knowledge2halfshift[idx] - val))
        if incl_shifts:
            if "penalty_vars_shifts" not in prev_solution:
                penalty_vars_shifts = intvar(0, 1000, shape=(num_operators, num_shifts - 2), name="penalty_vars_shifts")
                varmap["penalty_vars_shifts"] = penalty_vars_shifts
            else:    
                for idx, val in enumerate(prev_solution["penalty_vars_shifts"]):
                    soft_warm_start_obj += (abs(penalty_vars_shifts[idx] - val))
                    #warm_start_constraints.append(penalty_vars_shifts[idx] == val)
            #is machine used in the a shift?
            machine_used = boolvar(shape=num_machines,name="machine_used")
            varmap["machine_used"] = machine_used
            #unique machines?
            #num_unique_machines = intvar(0, num_machines,name="num_unique_machines")
        if incl_switch:    
            if "penalty_vars_switch" not in prev_solution:
                penalty_vars_switch = intvar(0, 1000, shape=(num_operators, num_shifts), name="penalty_vars_switch")
                varmap["penalty_vars_switch"] = penalty_vars_switch
            else:            
                for idx, val in enumerate(prev_solution["penalty_vars_switch"]):
                    soft_warm_start_obj += (abs(penalty_vars_switch[idx] - val))
                    #warm_start_constraints.append(penalty_vars_switch[idx] == val)
        if incl_ergo:
              if "penalty_vars_ergo" not in prev_solution:
                penalty_vars_ergo = intvar(0, 1000, shape=(num_operators, num_shifts), name="penalty_vars_ergo")
                varmap["penalty_vars_ergo"] = penalty_vars_ergo
              else:            
                for idx, val in enumerate(prev_solution["penalty_vars_ergo"]):
                    soft_warm_start_obj += (abs(penalty_vars_ergo[idx] - val))
                    #warm_start_constraints.append(penalty_vars_ergo[idx] == val)
        if incl_allocation:    
            if "penalty_vars_allocation" not in prev_solution:
                 penalty_vars_allocation = intvar(0, 1000, shape=(num_operators, num_shifts), name="penalty_vars_allocation")
                 varmap["penalty_vars_allocation"] = penalty_vars_allocation
            else:            
                for idx, val in enumerate(prev_solution["penalty_vars_allocation"]):
                    soft_warm_start_obj += (abs(penalty_vars_allocation[idx] - val))
                    #warm_start_constraints.append(penalty_vars_allocation[idx] == val)
            total_operator_time = intvar(0, shift_duration, shape=(num_operators, num_shifts,2), name="total_operator_time")

    initend_time = time.time()
    initialization_time = initend_time - initstart_time 
    print("TIME FOR INITIALIZATION: ", initialization_time)
    print("MODEL for ITERATION: ",iteration)
    #HARD CONSTRAINTS
    model = Model()
    #CONSTRAINT the first task should start at timestep == 0
    model += (min(start_time) == 0)

               
    # CONSTRAINT
    # the end time of a task equals the start time + execution time
    # exetime depends on (operator, machine and knowledge level of the operator for the machine)
    # Set execution time based on operator and machine combinations
    for i in range_operations:
        model += sum(#sum works as logical or --> At least one valid (machine, operator) combination
            (flow_machine2job[i] == m) & 
            (operator_knowl[o, m] > 0) & 
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

            for s in range(num_shifts):

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
                # #track operator knowledge
                # model += (flow_operator2halfshift[i, s, h] != -1).implies(
                #     #(flow_knowledge2halfshift[i, s, h] == operator_knowl[flow_operator2halfshift[i, s, h],flow_machine2job[i]])
                #     (flow_knowledge2halfshift[i, s, h] == 1)
                #     )  
                # model += (flow_operator2halfshift[i, s, h] == -1).implies(
                #     (flow_knowledge2halfshift[i, s, h] == 0)
                #     )
               #a if no overlap, than the task is not assigned to the half shift
                model += (flow_tasks2halfshift[i, s, h] == 0).implies(
                    (start_time[i] >= halfshift_end) | (end_time[i] <= halfshift_start)
                    )    
                model += (flow_operator2halfshift[i, s, h] == -1).implies(
                    (flow_tasks2halfshift[i, s, h] == 0)
                    )

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
    
    if iteration > 0:
        if incl_shifts:
            #SOFT CONSTRAINT: consecutive shifts: if operator works in shift 0, he cannot work in shifts -2,-1,1,2 
                
                flow_operator2halfshift = varmap["flow_operator2halfshift"]
                penalty_vars_shifts = varmap["penalty_vars_shifts"]
                
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
                            #constr1 = (sum(flow_operator2halfshift[:, s, :] == o) > 0).implies(penalty_vars_shifts[o, s] == pen_weight_shifts * shift_violations_plus1)
                            #constr2 = (sum(flow_operator2halfshift[:, s, :] == o) > 0).implies(penalty_vars_shifts[o, s] == int(pen_weight_shifts/2) * shift_violations_plus2)
                            
                            #constraints.append(constr1)
                            #constraints.append(constr2)
        
        if incl_ergo:
            flow_operator2halfshift = varmap["flow_operator2halfshift"]
            penalty_vars_ergo = varmap["penalty_vars_ergo"]
            start_time = varmap["start_time"]
            end_time = varmap["end_time"]
            
            for s in range(num_shifts):
                start_shift = shift_duration * s               
                end_shift = start_shift + shift_duration
    
                for i in range_operations:
                    ergo_overlap_time = max(0, min(end_shift, end_time[i]) - max(start_shift, start_time[i]))
    
                    penalty_value = ((ergo_overlap_time > 4) & (flow_operator2halfshift[i, s, 0] == flow_operator2halfshift[i, s, 1])) * (ergo_overlap_time - ub_ergo) * pen_weight_ergo
    
                    for o in range_operators:
                        model += (penalty_vars_ergo[o, s] == penalty_value)
                        
        if incl_allocation:
            flow_operator2halfshift = varmap["flow_operator2halfshift"]
            penalty_vars_allocation = varmap["penalty_vars_allocation"]
            start_time = varmap["start_time"]
            end_time = varmap["end_time"]
            total_operator_time = varmap["total_operator_time"]
            
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
                    model += ((total_operator_time[o, s, 0] > 0) & (total_operator_time[o, s, 1] == 0) & (total_operator_time[o, s, 0] < 3)).implies(
                        penalty_vars_allocation[o, s] == pen_weight_allocation * (lb_allocation_half - total_operator_time[o, s, 0])
                        )
                    model += ((total_operator_time[o, s, 1] > 0) & (total_operator_time[o, s, 0] == 0) & (total_operator_time[o, s, 1] < 3)).implies(
                        penalty_vars_allocation[o, s] == pen_weight_allocation * (lb_allocation_half - total_operator_time[o, s, 1])
                        )
                           
                    # check the shift, put pen
                    model += ((total_operator_time[o, s, 0] > 0) & (total_operator_time[o, s, 1] > 0) & (sum(total_operator_time[o, s, :] for h in range(2)) < lb_allocation)).implies(
                        penalty_vars_allocation[o, s] == pen_weight_allocation * (lb_allocation- sum(total_operator_time[o, s, :] for h in range(2)))
                    )
                    
        if incl_switch:
              flow_operator2halfshift = varmap["flow_operator2halfshift"]
              flow_machine2job = varmap["flow_machine2job"]
              penalty_vars_switch = varmap["penalty_vars_switch"]
              machine_used = varmap["machine_used"]
              
              for o in range_operators:
                  for s in range(num_shifts):
                      #make a list of machines, initialized to 0 = not used
                      
                      for i in range(num_operations):
                          for h in range(2):  
                              if flow_operator2halfshift[i, s, h] == o:
                                  model += (machine_used[flow_machine2job[i]] == 1)
                              else:    
                                  model += (machine_used[flow_machine2job[i]] == 0)
                      #for the operator in the shift, count the number of machines
                      #num_unique_machines = sum(machine_used)  
                      
                      # set the penalty on the switches
                      model += penalty_vars_switch[o, s] == (max(0, sum(machine_used)  - 1) * pen_weight_switch)      


    print('================SOLVING CP====================')
    if main_obj == 0:
        objective_makespan = max(0,(max(varmap["end_time"]) - int(desired_makespan)))
    elif main_obj == 1:
        objective_makespan = max(0,(max(varmap["end_time"]) - int(desired_makespan))**2)
    elif main_obj == 2:
        objective_makespan = max(0,(max(varmap["end_time"]) - int(desired_makespan))**3)
    elif main_obj == 3:
        objective_makespan =(max(varmap["end_time"]) - int(desired_makespan))
    elif main_obj == 4:
        objective_makespan =(max(varmap["end_time"]) - int(desired_makespan))**2 
    elif main_obj == 5:
        objective_makespan =(max(varmap["end_time"]) - int(desired_makespan))**3
    
    multi_objective = pen_weight_makespan * objective_makespan + soft_warm_start_obj 
    
    if iteration > 0:
        if incl_allocation:
            obj_allocation = sum(varmap["penalty_vars_allocation"])
            multi_objective += pen_weight_allocation * obj_allocation
        if incl_ergo:
           obj_ergo = sum(varmap["penalty_vars_ergo"])
           multi_objective += pen_weight_ergo * obj_ergo
        if incl_knowledge:
           obj_knowledge = sum(varmap["penalty_vars_knowledge"])
           multi_objective += pen_weight_knowledge * obj_knowledge
        if incl_shifts:
           obj_shifts = sum(varmap["penalty_vars_shifts"])
           multi_objective += pen_weight_shifts * obj_shifts
        if incl_switch:
           obj_switch = sum(varmap["penalty_vars_switch"])
           multi_objective += pen_weight_switch * obj_switch 
     
    model.minimize(multi_objective) 
    
    optim_info = {
       "makespan": objective_makespan,
       "weighted_makespan": pen_weight_makespan * objective_makespan,
       "total_objective": multi_objective,
       "sof_warm_start": soft_warm_start_obj
           }    
    if iteration > 0:
        if incl_allocation:
            optim_info["weighted_allocation"] = pen_weight_allocation * obj_allocation
            optim_info["allocation"] = obj_allocation
        if incl_ergo:
            optim_info["weighted_ergo"] = pen_weight_ergo * obj_ergo
            optim_info["ergo"] = obj_ergo
        if incl_knowledge:
            optim_info["weighted knowledge"] = pen_weight_knowl * obj_knowledge
            optim_info["knowledge"] = obj_knowledge 
        if incl_shifts:
            optim_info["weighted_shift"] = pen_weight_shifts * obj_shifts
            optim_info["shift"] = obj_shifts
        if incl_switch:
            optim_info["weighted_switch"] = pen_weight_switch * obj_switch
            optim_info["switch"] = obj_switch
    
    
    s = cp.SolverLookup.get("gurobi", model) #or gurobi
    val = s.solve()
    #val = model.solve(solver=solver)
    print(val)
    print(s.status())
    if s.solve():
        print("Solution FOUND")
    
        prev_solution = {k: v.value() for k, v in varmap.items()}
    
        
        # Get individual contributions
        print("Updated Objective Contributions:")
        print(f"  - Makespan Contribution: {optim_info['makespan'].value()}")
        #print(f"  - Knowledge Contribution: {objective_info['knowledge'].value()}")
        print(f"  - Weighted Makespan: {optim_info['weighted_makespan'].value()}")
        #print(f"  - Weighted Knowledge: {objective_info['weighted_knowledge'].value()}")
        print(f"  - Total Objective: {optim_info['total_objective'].value()}")
        
        if iteration > 0:
            print(f"  - Soft warm start Objective: {optim_info['soft_warm_start_obj'].value()}")                                                                                                                                                                                          
            if incl_allocation:
                print(f"  - Weighted ALLOCATION: {optim_info['weighted_allocation'].value()}")
                print(f"  -  Shift: {optim_info['allocation'].value()}")
            if incl_ergo:
                print(f"  - Weighted ERGO: {optim_info['weighted_ergo'].value()}")
                print(f"  -  Shift: {optim_info['ergo'].value()}")
            if incl_knowledge:
                print(f"  - Weighted KNOWLEDGE: {optim_info['weighted_knowledge'].value()}")
                print(f"  -  Shift: {optim_info['knowledge'].value()}")
            if incl_shifts:
                print(f"  - Weighted Shifts: {optim_info['weighted_shift'].value()}")
                print(f"  -  Shift: {optim_info['shift'].value()}")
            if incl_switch:
                print(f"  - Weighted SWITCH: {optim_info['weighted_switch'].value()}")
                print(f"  -  Shift: {optim_info['switch'].value()}")
            
        flow_operator2halfshift = varmap["flow_operator2halfshift"]
        flow_tasks2halfshift = varmap["flow_tasks2halfshift"]
        flow_machine2job = varmap["flow_machine2job"]
        end_time = varmap["end_time"]
        start_time = varmap["start_time"]
        plot_shift_schedule(start_time.value(), end_time.value(), flow_machine2job.value(), flow_operator2halfshift.value(), flow_tasks2halfshift.value(), num_operations, num_machines, num_operators, shift_duration, num_shifts)
        iteration += 1
        #num_empty_shifts = detect_empty_shifts(flow_tasks2halfshift.value())
        #print("EMPTY SHIFTS: ",num_empty_shifts)
        # if num_empty_shifts > 2:
        #     num_shifts -= 1
        #     desired_makespan = shift_duration*num_shifts
        #     print('DESIRED MAKESPAN: ',desired_makespan)
    else:
        print("No feasible solution found.")
        iteration = 9999
    
  
            

