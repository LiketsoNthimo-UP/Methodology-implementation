# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 01:48:59 2020

@author: nthimol01
"""

import pandas as pd
import numpy as np
import math
import random
from datetime import datetime
import multiprocessing

def cooling_schedule(t, initial_temperature = 1000, phi = 0.9):
    current_temperature = initial_temperature * phi ** t
    return current_temperature

def accept_bad_solution(delta_cost, current_temperature):
    decision = False
    decision = np.exp(-1 * delta_cost / current_temperature)
    return decision

def calculate_cost(shift_schedule):
    rows = shift_schedule.shape[0]
    total_cost = 0
    
    for i in range(rows):
        shift_start = shift_schedule.iloc[i,0]
        shift_end = shift_schedule.iloc[i,1]
        shift_break_start = shift_schedule.iloc[i,2]
        shift_break_end = shift_schedule.iloc[i,3]
        shift_staff_number = shift_schedule.iloc[i,4]
        
        shift_cost = (shift_end - shift_start - (shift_break_end - shift_break_start)) * shift_staff_number
        total_cost = total_cost + shift_cost
    
    return total_cost

def generate_agent_logs(shift_schedule):
    rows = shift_schedule.shape[0]
    agent_logs = pd.DataFrame()
    
    for i in range(rows):
        number_of_agents_in_shift = shift_schedule.iloc[i,4]
        for j in range(number_of_agents_in_shift):
            agent_logs = agent_logs.append(shift_schedule.iloc[i,0:4], sort=False)
    
    agent_logs = agent_logs[['Start_time', 'End_time', 'Break_start', 'Break_end']]
    agent_logs = agent_logs.sort_values(by = ['Start_time', 'End_time', 'Break_start', 'Break_end'])
    agent_logs = agent_logs.reset_index(drop = True)
    return agent_logs
    

def update_break_periods(agent_logs, arrival_time):
    agent_logs = agent_logs.copy()
    agent_logs_on_break = agent_logs.loc[(agent_logs['Break_start'] <= arrival_time) & (agent_logs['Break_end'] > arrival_time),]
    agent_logs_on_shift = agent_logs.loc[(agent_logs['Break_start'] > arrival_time) | (agent_logs['Break_end'] <= arrival_time),]
    agent_logs_on_break = agent_logs_on_break.assign(Start_time = agent_logs_on_break['Break_end'])
    agent_logs = agent_logs_on_break.append(agent_logs_on_shift)
    return agent_logs


def update_agent_times(agent_logs, call_features, agent_index):
    agent_logs = agent_logs.copy()
    arrival_time = call_features.iloc[0,0]
    agent_logs = update_break_periods(agent_logs, arrival_time)
    
    agent_logs.iloc[agent_index,0] = agent_logs.iloc[agent_index,0] + call_features.iloc[0,1]
    if agent_logs.iloc[agent_index,0] > agent_logs.iloc[agent_index,1]:
        #agent_logs = agent_logs.iloc[1:,]
        agent_logs = agent_logs.drop(agent_logs.index[agent_index])
        
    agent_logs = agent_logs.sort_values(by = ['Start_time', 'End_time', 'Break_start', 'Break_end'])
    agent_logs = agent_logs.reset_index(drop = True)
    return agent_logs

def check_shift_feasibility(shift_schedule, call_centre_scenario, max_wait):
    call_centre_scenario_operation = pd.DataFrame()
    unique_days = call_centre_scenario['Day'].unique()
    initial_agent_logs = generate_agent_logs(shift_schedule)
    is_feasible = True
    service_level = 0.8
    
    for day in unique_days:
        day_scenario = call_centre_scenario[call_centre_scenario['Day'] == day]
        agent_logs = initial_agent_logs.copy()
        
        rows = day_scenario.shape[0]
        for i in range(rows):
            call_features = day_scenario[i:i+1]
            
            if agent_logs.shape[0] > 0:
                agent_rows = agent_logs.shape[0]
                #print('agent rows: ',agent_rows)
                for j in range(agent_rows):
                    # Checks that call can be answered before customer runs out of patience and that call arrives before agent shift has ended
                    # also check that the call arrives before the agent leaves for break or that the agent is back before the customer drops
                    if (agent_logs.iloc[j,0] <= call_features.iloc[0,0] + call_features.iloc[0,2]) and (agent_logs.iloc[j,1] > call_features.iloc[0,0]) and ((agent_logs.iloc[j,2] > call_features.iloc[0,0]) or (agent_logs.iloc[j,3] <= call_features.iloc[0,0] + call_features.iloc[0,2])):
                        call_features = call_features.assign(Waiting_time = np.maximum(agent_logs.iloc[0,0] - call_features.iloc[0,0],0))
                        call_features = call_features.assign(Call_answered = 'yes')
                        call_centre_scenario_operation = call_centre_scenario_operation.append(call_features)
                        agent_logs = update_agent_times(agent_logs, call_features, j)
                        break
                    
                if call_features.shape[1] != 7:
                    call_features = call_features.assign(Waiting_time = call_features.iloc[0,2])
                    call_features = call_features.assign(Call_answered = 'no')
                    call_centre_scenario_operation = call_centre_scenario_operation.append(call_features)
                
            else:
                call_features = call_features.assign(Waiting_time = call_features.iloc[0,2])
                call_features = call_features.assign(Call_answered = 'no')
                call_centre_scenario_operation = call_centre_scenario_operation.append(call_features)
    
    for performance_interval in range(1,145):
        performance_dataset = call_centre_scenario_operation[call_centre_scenario_operation['Performance_interval'] == performance_interval]
        service_level_dataset = performance_dataset[performance_dataset['Waiting_time'] <= max_wait]
        all_calls = performance_dataset.shape[0]
        service_level_calls = service_level_dataset.shape[0]
        
        if all_calls == 0:
            continue
        
        if service_level_calls / all_calls < service_level:
            is_feasible = False
            break
        
    return is_feasible
    

def initial_shift_set(shifts, call_centre_scenario, max_wait):
    rows = shifts.shape[0]
    
    staff_number = 1
    staff_number_list = [1] * rows
    shift_schedule = shifts.copy()
    shift_schedule['Staff_number'] = staff_number_list
    
    is_feasible = check_shift_feasibility(shift_schedule, call_centre_scenario, max_wait)
    
    while not is_feasible:
        staff_number = staff_number + 1
        staff_number_list = [staff_number] * rows
        shift_schedule['Staff_number'] = staff_number_list
        is_feasible = check_shift_feasibility(shift_schedule, call_centre_scenario, max_wait)
    
    return shift_schedule
    
def simulated_annealing(shifts, call_centre_scenario, max_wait,  max_iteration = 10):
    start = datetime.now()
    w_init = initial_shift_set(shifts, call_centre_scenario, max_wait)
    cost_init = calculate_cost(w_init)
    w_best = w_init.copy()
    cost_best = cost_init.copy()
    shift_count = shifts.shape[0]
    
    best_fitness_value_history = [] # Best fitness function we have seen
    explored_fitness_value_history = [] # All fitness functions we have accepted (whether good or random)
    
    for t in range(max_iteration):
        print('time at iteration ',t , ': ', datetime.now() - start)
        T = cooling_schedule(t)
        if T == 0:
            return w_best, best_fitness_value_history, explored_fitness_value_history
        
        delta_w = random.choices([-1, -1, 0, 1], k = shift_count)
        w_next = w_best.copy()
        w_next['Staff_number'] = w_next['Staff_number'] + delta_w
        w_next['Staff_number'] = w_next['Staff_number'].replace([-1],0)
        cost_next = calculate_cost(w_next)
        
        if any(w_next['Staff_number']<0):
            continue
        
        explored_fitness_value_history.append(cost_next)
        
        delta_cost = cost_next - cost_best
        
        if cost_best > cost_next or accept_bad_solution(delta_cost, T):
            is_feasible = check_shift_feasibility(w_next, call_centre_scenario, max_wait)
            
            if is_feasible:
                w_best = w_next.copy()
                cost_best = cost_next.copy()
                best_fitness_value_history.append(cost_best)
            
            else:
                continue
        
    return w_best, best_fitness_value_history, explored_fitness_value_history


def run_test_case(service_rate, offered_load, planning_period, sample_size):
    shifts = pd.read_csv('../Part 1 datasets/Shifts.csv')
    
    service_rate = str(service_rate)
    offered_load = str(offered_load)
    RA = '1.0'
    abandonment_rate = str(service_rate)
    max_wait = 0
    
    scenario_name = '../Part 1 datasets/Scenario_' + service_rate + '_' + offered_load + '_' + RA + '_' + abandonment_rate + '.csv'
    
    call_centre_scenario = pd.read_csv(scenario_name)
    call_centre_scenario = call_centre_scenario.assign(Performance_interval = (call_centre_scenario['Arrival_time']/(60*5)).apply(np.ceil))
    call_centre_scenario['Arrival_time'] = call_centre_scenario['Arrival_time']/60/60
    call_centre_scenario['Service_duration'] = call_centre_scenario['Service_duration']/60/60
    call_centre_scenario['Time_to_abandon'] = call_centre_scenario['Time_to_abandon']/60/60
    
    shifts = shifts[shifts['Planning_period'] == planning_period]
    del shifts['Planning_period']
    shifts = shifts.reset_index(drop = True)
    
    call_centre_scenario_minimized = call_centre_scenario[call_centre_scenario.Day.isin(range(1,sample_size))]
    best_shift, best_history, explored_history = simulated_annealing(shifts, call_centre_scenario_minimized, max_wait, max_iteration = 250)
    
    df_best_shift = pd.DataFrame(best_shift)
    df_best_history = pd.DataFrame(best_history)
    df_explored_history = pd.DataFrame(explored_history)
    df_explored_history.columns = ['Explored_history']
    df_best_history.columns = ['Best_history']
    
    consolidation = pd.concat([df_explored_history, df_best_history.reindex(df_explored_history.index)], axis=1)
    consolidation = pd.concat([consolidation, df_best_shift.reindex(consolidation.index)], axis=1)
    return consolidation


test_cases = pd.read_csv('../Part 1 datasets/Test_cases.csv')

test_cases = test_cases[test_cases['Service_rate'] == 2]
test_cases = test_cases[test_cases['Offered_load'] == 32]
test_cases = test_cases[test_cases['Planning_period'] == 1]
test_cases = test_cases.reset_index(drop = True)
rows = test_cases.shape[0]

for i in range(rows):
    service_rate = test_cases.iloc[i,0]
    offered_load = test_cases.iloc[i,1]
    planning_period = test_cases.iloc[i,2]
    
    if offered_load == 16:
        test_case_results = run_test_case(service_rate, offered_load, planning_period, 29)
        file_name = 'Results_' + str(service_rate) + '_' + str(offered_load) + '_' + str(planning_period) + '_sample28.csv'
        test_case_results.to_csv(file_name, index=False)
        
    elif offered_load == 32:
        test_case_results = run_test_case(service_rate, offered_load, planning_period, 15)
        file_name = 'Results_' + str(service_rate) + '_' + str(offered_load) + '_' + str(planning_period) + '_sample14.csv'
        test_case_results.to_csv(file_name, index=False)
        
    elif offered_load == 64:
        test_case_results = run_test_case(service_rate, offered_load, planning_period, 8)
        file_name = 'Results_' + str(service_rate) + '_' + str(offered_load) + '_' + str(planning_period) + '_sample7.csv'
        test_case_results.to_csv(file_name, index=False)