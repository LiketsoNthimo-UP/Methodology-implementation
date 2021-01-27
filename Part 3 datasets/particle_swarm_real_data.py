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
    agent_logs = agent_logs.sort_index()
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

def good_performance_periods(shift_schedule, call_centre_scenario, max_wait):
    call_centre_scenario_operation = pd.DataFrame()
    unique_days = call_centre_scenario['Day'].unique()
    print(unique_days)
    initial_agent_logs = generate_agent_logs(shift_schedule)
    service_level = 0.8
    total_met_sl_periods = 0
    
    for day in unique_days:
        print(day)
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
            service_level_dataset = performance_dataset[performance_dataset['Waiting_time'] <= max_wait]
            all_calls = performance_dataset.shape[0]
            service_level_calls = service_level_dataset.shape[0]
            
            if all_calls == 0:
                total_met_sl_periods = total_met_sl_periods + 1
            
            else:
                if service_level_calls / all_calls >= service_level:
                    total_met_sl_periods = total_met_sl_periods + 1
        
        print('day: ' + str(day) + ' with ', str(total_met_sl_periods), ' good periods.')
        
    return total_met_sl_periods
    

def initial_shift_set(shifts, call_centre_scenario, max_wait, max_agents):
    rows = shifts.shape[0]
    
    staff_number_list = [0] * rows
    shift_schedule = shifts.copy()
    shift_schedule['Staff_number'] = staff_number_list
    
    cycles = int(max_agents / rows) + 1
    assigned_agents = 0
    
    for i in range(cycles):
        for j in range(rows):
            shift_schedule.iloc[j,4] = shift_schedule.iloc[j,4] + 1
            assigned_agents = assigned_agents + 1
            
            if assigned_agents >= max_agents:
                break
        
        if assigned_agents >= max_agents:
                break
    
    return shift_schedule
    
def particle_swarm_optimisation(shifts, call_centre_scenario, max_wait, max_cost, max_agents, max_iteration = 10):
    start = datetime.now()
    w_init = initial_shift_set(shifts, call_centre_scenario, max_wait, max_agents)
    shift_count = shifts.shape[0]
    
    particle_count = 8
    particles = pd.DataFrame()
    velocities = pd.DataFrame()
    cost_per_particle = pd.DataFrame()
    good_periods_per_particle = pd.DataFrame()
    
    for i in range(particle_count):
        particles[i] = w_init['Staff_number'] + random.choices([0, -1], k = shift_count)
        particles[i] = particles[i].replace([-1],0)
        velocities[i] = [0] * shift_count
        
        particle_shift = w_init.copy()
        particle_shift['Staff_number'] = particles[i]
        particle_cost = calculate_cost(particle_shift)
        particle_good_periods = good_performance_periods(particle_shift, call_centre_scenario, max_wait)
        print(particle_good_periods)
        cost_per_particle[i] = [particle_cost]
        good_periods_per_particle[i] = [particle_good_periods]
    
    particles_best = particles.copy()
    
    best_particle_index_series = good_periods_per_particle.idxmax(axis = 1)
    best_particle_index = best_particle_index_series[0]
    
    swarm_best = particles[best_particle_index]
    swarm_best_good_periods = good_periods_per_particle.iloc[0, best_particle_index]
    
    best_fitness_value_history = [] # Best fitness function we have seen
    explored_fitness_value_history = [] # All fitness functions we have accepted (whether good or random)
    
    for t in range(max_iteration):
        print('time at iteration ',t , ': ', datetime.now() - start)
        
        for i in range(particle_count):
            r_particle = random.choices([0, 1], k = shift_count)
            r_swarm = random.choices([0, 1], k = shift_count)
            velocities[i] = velocities[i] + (particles_best[i] - particles[i]) * r_particle + (swarm_best - particles[i]) * r_swarm
            particles[i] = particles[i] + velocities[i]
            particles[i][particles[i] < 0] = 0
            
            current_particle_shift = w_init.copy()
            current_particle_shift['Staff_number'] = particles[i]
            current_particle_cost = calculate_cost(current_particle_shift)
            current_particle_agents = current_particle_shift['Staff_number'].sum()
            
            if current_particle_cost > max_cost:
                continue
        
            if current_particle_agents > max_agents:
                continue
            
            current_particle_good_periods = good_performance_periods(current_particle_shift, call_centre_scenario, max_wait)
            explored_fitness_value_history.append(current_particle_good_periods)
            
            if current_particle_good_periods > good_periods_per_particle.iloc[0,i]:
                particles_best[i] = particles[i]
                good_periods_per_particle.iloc[0,i] = current_particle_good_periods
                
                if current_particle_good_periods > swarm_best_good_periods:
                    swarm_best = particles_best[i]
                    swarm_best_good_periods = current_particle_good_periods.copy()
                    best_fitness_value_history.append(swarm_best_good_periods)
            
        w_best = w_init.copy()
        w_best['Staff_number'] = swarm_best
        
    return w_best, best_fitness_value_history, explored_fitness_value_history


def run_test_case(average_time_to_abandon, sample_size):
    shifts = pd.read_csv('../Part 3 datasets/Shifts.csv')
    
    average_time_to_abandon = str(average_time_to_abandon)
    max_wait_sec = 20
    max_wait = max_wait_sec/60/60
    max_cost = 414
    max_agents = 57
    
    scenario_name = 'individual_arrivals_' + str(average_time_to_abandon) + '.csv'
    
    call_centre_scenario = pd.read_csv(scenario_name)
    call_centre_scenario = call_centre_scenario.assign(Performance_interval = (call_centre_scenario['Arrival_time']/(60*15)).apply(np.ceil))
    call_centre_scenario['Arrival_time'] = call_centre_scenario['Arrival_time']/60/60
    call_centre_scenario['Service_duration'] = call_centre_scenario['Service_duration']/60/60
    call_centre_scenario['Time_to_abandon'] = call_centre_scenario['Time_to_abandon']/60/60
    
    call_centre_scenario_minimized = call_centre_scenario[call_centre_scenario.Day.isin(range(1,sample_size))]
    best_shift, best_history, explored_history = particle_swarm_optimisation(shifts, call_centre_scenario_minimized, max_wait, max_cost, max_agents, max_iteration = 4)
    
    df_best_shift = pd.DataFrame(best_shift)
    df_best_history = pd.DataFrame(best_history)
    df_explored_history = pd.DataFrame(explored_history)
    df_explored_history.columns = ['Explored_history']
    '''df_best_history.columns = ['Best_history']
    
    consolidation = pd.concat([df_explored_history, df_best_history.reindex(df_explored_history.index)], axis=1)
    consolidation = pd.concat([consolidation, df_best_shift.reindex(consolidation.index)], axis=1)
    return consolidation'''


average_time_to_abandon = 30
test_case_results = run_test_case(average_time_to_abandon, 2)
#file_name = 'Results_' + str(average_time_to_abandon) + '.csv'
#test_case_results.to_csv(file_name, index=False)
