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
        shift_start = shift_schedule.iloc[i,1]
        shift_end = shift_schedule.iloc[i,2]
        shift_break_start = shift_schedule.iloc[i,3]
        shift_break_end = shift_schedule.iloc[i,4]
        shift_staff_number = shift_schedule.iloc[i,5]
        
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

def populate_performance_info(shift_schedule_per_day, call_centre_scenario, max_wait):
    call_centre_scenario_operation = pd.DataFrame()
    per_day_performance_info = pd.DataFrame()
    unique_days = call_centre_scenario['Day'].unique()
    service_level = 0.8
    total_met_sl_periods = 0
    total_cost = 0
    total_answered_calls = 0
    total_dropped_calls = 0
    start = datetime.now()
    
    for day in unique_days:
        day_shift_schedule = shift_schedule_per_day[shift_schedule_per_day['Day'] == day].copy()
        day_cost = calculate_cost(day_shift_schedule)
        del day_shift_schedule['Day']
        day_shift_schedule = day_shift_schedule.reset_index(drop = True)
        
        day_answered_calls = 0
        day_dropped_calls = 0
        day_good_periods = 0

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
                        day_answered_calls = day_answered_calls + 1
                        total_answered_calls = total_answered_calls + 1
                        break
                    
                if call_features.shape[1] != 7:
                    call_features = call_features.assign(Waiting_time = call_features.iloc[0,2])
                    call_features = call_features.assign(Call_answered = 'no')
                    call_centre_scenario_operation = call_centre_scenario_operation.append(call_features)
                    day_dropped_calls = day_dropped_calls + 1
                    total_dropped_calls = total_dropped_calls + 1
                
            else:
                call_features = call_features.assign(Waiting_time = call_features.iloc[0,2])
                call_features = call_features.assign(Call_answered = 'no')
                call_centre_scenario_operation = call_centre_scenario_operation.append(call_features)
                day_dropped_calls = day_dropped_calls + 1
                total_dropped_calls = total_dropped_calls + 1
    
        for performance_interval in range(1,73):
            performance_dataset = call_centre_scenario_operation[call_centre_scenario_operation['Performance_interval'] == performance_interval]
            all_eligible_calls_dataset = performance_dataset.loc[(performance_dataset['Waiting_time'] >= 5/60/60) | (performance_dataset['Call_answered'] == 'yes')]
            service_level_dataset = performance_dataset.loc[(performance_dataset['Waiting_time'] <= max_wait) & (performance_dataset['Call_answered'] == 'yes')]
            all_calls = all_eligible_calls_dataset.shape[0]
            service_level_calls = service_level_dataset.shape[0]
            
            if all_calls == 0:
                total_met_sl_periods = total_met_sl_periods + 1
                day_good_periods = day_good_periods + 1
            
            else:
                if service_level_calls / all_calls >= service_level:
                    total_met_sl_periods = total_met_sl_periods + 1
                    day_good_periods = day_good_periods + 1
        
        total_cost = total_cost + day_cost
        append_day_features = np.zeros((1,5))
        
        append_day_features[0,0] = day
        append_day_features[0,1] = day_good_periods
        append_day_features[0,2] = day_cost
        append_day_features[0,3] = day_answered_calls
        append_day_features[0,4] = day_dropped_calls
        
        per_day_performance_info = per_day_performance_info.append(pd.DataFrame(append_day_features))
        print('Day ',day , 'appended at: ', datetime.now() - start)
    
    append_day_features = np.zeros((1,4))
    
    append_day_features[0,0] = total_met_sl_periods
    append_day_features[0,1] = total_cost
    append_day_features[0,2] = total_answered_calls
    append_day_features[0,3] = total_dropped_calls

    total_performance_info = pd.DataFrame(append_day_features)

    return pd.concat([per_day_performance_info, total_performance_info], axis = 1)
    

'''
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
'''


sample_size = 3
day_range = range(1,sample_size)

max_wait_sec = 20
max_wait = max_wait_sec/60/60

average_time_to_abandon = 30
average_time_to_abandon = str(average_time_to_abandon)

shift_file_path = '../Part 3 datasets/Shifts_actual.csv'
shift_info = pd.read_csv(shift_file_path)
shift_info = shift_info[['Day','Start_time','End_time','Break_start','Break_end','Staff_number']]
shift_info = shift_info[shift_info['Staff_number'] > 0]
shift_info.Staff_number = shift_info.Staff_number.astype(int)
shift_info = shift_info.reset_index(drop = True)

calls_file_path = '../Part 3 datasets/individual_arrivals_' +  average_time_to_abandon + '.csv'
call_centre_scenario = pd.read_csv(calls_file_path)
call_centre_scenario = call_centre_scenario[call_centre_scenario.Day.isin(day_range)]
call_centre_scenario = call_centre_scenario.assign(Performance_interval = (call_centre_scenario['Arrival_time']/(60*15)).apply(np.ceil))
call_centre_scenario['Arrival_time'] = call_centre_scenario['Arrival_time']/60/60
call_centre_scenario['Service_duration'] = call_centre_scenario['Service_duration']/60/60
call_centre_scenario['Time_to_abandon'] = call_centre_scenario['Time_to_abandon']/60/60


performance_info = populate_performance_info(shift_info, call_centre_scenario, max_wait)
performance_info.columns = ['Day', 'Good_periods', 'Cost', 'Answered_calls', 'Dropped_calls', 'Total_good_periods', 'Total_cost', 'Total_answered_calls', 'Total_dropped_calls']
performance_info = performance_info.reset_index(drop = True)

file_name = 'Performance_info_actual_' + str(average_time_to_abandon) + '.csv'
performance_info.to_csv(file_name, index=False)