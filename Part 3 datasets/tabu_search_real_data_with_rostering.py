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

def calculate_cost(shift_schedule):
    total_cost = (shift_schedule['End_time'] - shift_schedule['Start_time'] - (shift_schedule['Break_end'] - shift_schedule['Break_start'])) * shift_schedule['Staff_number']
    total_cost = total_cost.sum()
    
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
    agent_logs = agent_logs.sort_index()
    return agent_logs


def update_agent_times(agent_logs, call_features, agent_index):
    agent_logs = agent_logs.copy()
    arrival_time = call_features.iloc[0,0]
    agent_logs = update_break_periods(agent_logs, arrival_time)
    
    agent_logs.iloc[agent_index,0] = arrival_time + call_features.iloc[0,1]
    agent_logs = agent_logs.loc[(agent_logs['Start_time'] < agent_logs['End_time']) & (agent_logs['End_time'] > arrival_time),]
        
    agent_logs = agent_logs.sort_values(by = ['Start_time', 'End_time', 'Break_start', 'Break_end'])
    agent_logs = agent_logs.reset_index(drop = True)
    return agent_logs

def good_performance_periods(shift_schedule_per_day, call_centre_scenario, max_wait, current_good_periods_shortage):
    call_centre_scenario_operation = pd.DataFrame()
    unique_days = call_centre_scenario['Day'].unique()
    service_level = 0.8
    total_met_sl_periods = 0
    days_observed = 0
    
    for day in unique_days:
        day_shift_schedule = shift_schedule_per_day[shift_schedule_per_day['Day'] == day].copy()
        del day_shift_schedule['Day']
        day_shift_schedule = day_shift_schedule.reset_index(drop = True)
        agent_logs = generate_agent_logs(day_shift_schedule)
        
        day_scenario = call_centre_scenario[call_centre_scenario['Day'] == day]
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
                        call_features = call_features.assign(Waiting_time = np.maximum(agent_logs.iloc[j,0] - call_features.iloc[0,0],0))
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
    
        for performance_interval in range(1,73):
            performance_dataset = call_centre_scenario_operation[call_centre_scenario_operation['Performance_interval'] == performance_interval]
            all_eligible_calls_dataset = performance_dataset.loc[(performance_dataset['Waiting_time'] >= 5/60/60) | (performance_dataset['Call_answered'] == 'yes')]
            service_level_dataset = performance_dataset.loc[(performance_dataset['Waiting_time'] <= max_wait) & (performance_dataset['Call_answered'] == 'yes')]
            all_calls = all_eligible_calls_dataset.shape[0]
            service_level_calls = service_level_dataset.shape[0]
            
            if all_calls == 0:
                total_met_sl_periods = total_met_sl_periods + 1
            
            else:
                if service_level_calls / all_calls >= service_level:
                    total_met_sl_periods = total_met_sl_periods + 1
        
        days_observed = days_observed + 1
        possible_good_periods_so_far = 72 * days_observed
        good_periods_shortage_so_far = possible_good_periods_so_far - total_met_sl_periods
        
        if good_periods_shortage_so_far > current_good_periods_shortage:
            print('Early interation cut-off on day: ' + str(day))
            break

        #print('day: ' + str(day) + ' with ', str(total_met_sl_periods), ' good periods.')
        
    return total_met_sl_periods
    

def initial_shift_set(shifts, max_agents):
    rows = shifts.shape[0]
    
    reordered_shifts = shifts.sort_values(by = ['Start_time', 'End_time', 'Cycle_number'])
    reordered_shifts = reordered_shifts.reset_index(drop = True)
    
    staff_number_list = [0] * rows
    shift_schedule = reordered_shifts.copy()
    shift_schedule['Staff_number'] = staff_number_list
    
    cycles = int(max_agents / rows) + 1
    assigned_agents = 0
    
    for i in range(cycles):
        for j in range(rows):
            shift_schedule.iloc[j,5] = shift_schedule.iloc[j,5] + 1
            assigned_agents = assigned_agents + 1
            
            if assigned_agents >= max_agents:
                break
        
        if assigned_agents >= max_agents:
                break
    
    shift_schedule = shift_schedule.sort_values(by = ['Cycle_number', 'Start_time', 'End_time'])
    shift_schedule = shift_schedule.reset_index(drop = True)
    
    return shift_schedule

def cycle_number_to_shift_day(shift_schedule, day_range):
    cycles = range(1,13)
    shift_schedule_per_day = pd.DataFrame(columns=['Day', 'Start_time', 'End_time', 'Break_start', 'Break_end', 'Staff_number'])
    
    for i in day_range:
        day_off_1 = [(j * -1 - 5 + i) % 12  for j in cycles]
        day_off_2 = [(j * -1 - 6 + i) % 12  for j in cycles]
        day_off_3 = [(j * -1 - 11 + i) % 12  for j in cycles]
        
        off_cycles = [day_off_1.index(0) + 1, day_off_2.index(0) + 1, day_off_3.index(0) + 1]
        
        day_schedule = shift_schedule[~shift_schedule.Cycle_number.isin(off_cycles)].copy()
        day_list = [i] * day_schedule.shape[0]
        day_schedule['Day'] = day_list
        day_schedule = day_schedule[['Day', 'Start_time', 'End_time', 'Break_start', 'Break_end', 'Staff_number']]
        
        shift_schedule_per_day = shift_schedule_per_day.append(day_schedule)
    
    shift_schedule_per_day = shift_schedule_per_day.reset_index(drop = True)
    
    return shift_schedule_per_day
    
def tabu_search(shifts, call_centre_scenario, max_wait, max_cost, max_agents, day_range, max_iteration = 10):
    start = datetime.now()
    w_init = initial_shift_set(shifts, max_agents)
    w_init_per_day = cycle_number_to_shift_day(w_init, day_range)
    possible_good_periods = 72 * len(day_range)
    good_periods_init = good_performance_periods(w_init_per_day, call_centre_scenario, max_wait, possible_good_periods)
    good_periods_best = good_periods_init
    w_best = w_init.copy()
    w_next = w_init.copy()

    init_best_candidate_list = []
    init_best_candidate_list = w_best['Staff_number'].copy()
    final_best_candidate_list = init_best_candidate_list.copy()
    tabu_list = []
    tabu_list.append(final_best_candidate_list.to_list())
    tabu_tenure = 4
    neighbour_count = 4
    neighbours = pd.DataFrame()
    current_candidate_list = []
    
    best_fitness_value_history = [] # Best fitness function we have seen
    explored_fitness_value_history = [] # All fitness functions we have accepted (whether good or random)
    best_fitness_value_history.append(good_periods_best)
    explored_fitness_value_history.append(good_periods_best)
    
    for t in range(max_iteration):
        print('time at iteration ',t , ': ', datetime.now() - start)
        
        if good_periods_best >= possible_good_periods:
            print('Service level met in all performance periods')
            break
 
        for i in range(neighbour_count):
            shift_has_members = [0 if i == 0 else 1 for i in init_best_candidate_list]
            shifts_to_reduce = [i * random.choice([1, 0]) for  i in shift_has_members]
            shifts_to_increase = shifts_to_reduce.copy()
            random.shuffle(shifts_to_increase)
            neighbours[i] = init_best_candidate_list + shifts_to_increase - shifts_to_reduce
        
        current_candidate_list = neighbours[0].copy()
        best_candidate_list = current_candidate_list.copy()
        w_next['Staff_number'] = best_candidate_list
        w_next_per_day = cycle_number_to_shift_day(w_next, day_range)
        cost_best_candidate = calculate_cost(w_next_per_day)
        candidates_assessed = 1

        good_periods_shortage = possible_good_periods
        good_periods_best_candidate = good_performance_periods(w_next_per_day, call_centre_scenario, max_wait, good_periods_shortage)
            
        while candidates_assessed < neighbour_count:
            
            if good_periods_best_candidate >= possible_good_periods:
                print('Service level met in all performance periods of candidate ' + str(candidates_assessed))
                break

            current_candidate_list = neighbours[candidates_assessed]
            w_next['Staff_number'] = current_candidate_list
            w_next_per_day = cycle_number_to_shift_day(w_next, day_range)
            cost_current_candidate = calculate_cost(w_next_per_day)

            if cost_current_candidate > max_cost:
                candidates_assessed = candidates_assessed + 1
                print('Candidate ', str(candidates_assessed), ' not feasible.')
                continue
            
            if any(w_next['Staff_number']<0):
                candidates_assessed = candidates_assessed + 1
                print('Candidate ', str(candidates_assessed), ' not feasible.')
                continue

            good_periods_current_candidate = good_performance_periods(w_next_per_day, call_centre_scenario, max_wait, good_periods_shortage)
            explored_fitness_value_history.append(good_periods_current_candidate)
            
            if cost_best_candidate > max_cost and not(current_candidate_list.to_list() in tabu_list):
                best_candidate_list = current_candidate_list
                cost_best_candidate = cost_current_candidate
                good_periods_best_candidate = good_periods_current_candidate
                candidates_assessed = candidates_assessed + 1
                good_periods_shortage = possible_good_periods - good_periods_current_candidate
                continue
            
            if good_periods_current_candidate > good_periods_best_candidate and not(current_candidate_list.to_list() in tabu_list):
                best_candidate_list = current_candidate_list
                cost_best_candidate = cost_current_candidate
                good_periods_best_candidate = good_periods_current_candidate
                candidates_assessed = candidates_assessed + 1
                good_periods_shortage = possible_good_periods - good_periods_current_candidate
                continue

            candidates_assessed = candidates_assessed + 1
        
        if cost_best_candidate <= max_cost:
            init_best_candidate_list = best_candidate_list.copy()
        
            if good_periods_best_candidate > good_periods_best:
                good_periods_best = good_periods_best_candidate
                final_best_candidate_list = init_best_candidate_list.copy()
                w_best['Staff_number'] = final_best_candidate_list
                best_fitness_value_history.append(good_periods_best)
        
        tabu_list.append(best_candidate_list.to_list())
        if len(tabu_list) > tabu_tenure:
            tabu_list = tabu_list[1:tabu_tenure].copy()

    return w_best, best_fitness_value_history, explored_fitness_value_history


def run_test_case(average_time_to_abandon, sample_size):
    shifts = pd.read_csv('../Part 3 datasets/Shifts_actual_with_cycle_number.csv')
    
    average_time_to_abandon = str(average_time_to_abandon)
    max_wait_sec = 20
    max_wait = max_wait_sec/60/60
    max_cost = 9329
    max_agents = 77
    
    scenario_name = 'individual_arrivals_' + str(average_time_to_abandon) + '.csv'
    
    call_centre_scenario = pd.read_csv(scenario_name)
    call_centre_scenario = call_centre_scenario.assign(Performance_interval = (call_centre_scenario['Arrival_time']/(60*15)).apply(np.ceil))
    call_centre_scenario['Arrival_time'] = call_centre_scenario['Arrival_time']/60/60
    call_centre_scenario['Service_duration'] = call_centre_scenario['Service_duration']/60/60
    call_centre_scenario['Time_to_abandon'] = call_centre_scenario['Time_to_abandon']/60/60
    
    day_range = range(1,sample_size)
    call_centre_scenario_minimized = call_centre_scenario[call_centre_scenario.Day.isin(day_range)]
    best_shift, best_history, explored_history = tabu_search(shifts, call_centre_scenario_minimized, max_wait, max_cost, max_agents, day_range, max_iteration = 63)
    
    df_best_shift = pd.DataFrame(best_shift)
    df_best_history = pd.DataFrame(best_history)
    df_explored_history = pd.DataFrame(explored_history)
    df_explored_history.columns = ['Explored_history']
    df_best_history.columns = ['Best_history']
    
    consolidation = pd.concat([df_explored_history, df_best_history], axis=1)
    consolidation = pd.concat([consolidation, df_best_shift], axis=1)
    return consolidation


average_time_to_abandon = 60
test_case_results = run_test_case(average_time_to_abandon, 24)
file_name = 'Results_TS_' + str(average_time_to_abandon) + '.csv'
test_case_results.to_csv(file_name, index=False)
