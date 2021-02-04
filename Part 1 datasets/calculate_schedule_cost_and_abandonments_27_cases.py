# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 01:48:59 2020

@author: nthimol01
"""

import pandas as pd
import numpy as np
import math
import random
import csv
from datetime import datetime

def export_data(dataset, filename):
    file_name = filename + '.csv'
    dataset.to_csv(file_name, index=False)

def calculate_cost(shift_schedule):
    total_cost = (shift_schedule['End_time'] - shift_schedule['Start_time'] - (shift_schedule['Break_end'] - shift_schedule['Break_start'])) * shift_schedule['Staff_number']
    total_cost = total_cost.sum()
    
    return total_cost/60/60

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
    if agent_logs.iloc[agent_index,0] > agent_logs.iloc[agent_index,1]:
        #agent_logs = agent_logs.iloc[1:,]
        agent_logs = agent_logs.drop(agent_logs.index[agent_index])
        
    agent_logs = agent_logs.sort_values(by = ['Start_time', 'End_time', 'Break_start', 'Break_end'])
    agent_logs = agent_logs.reset_index(drop = True)
    return agent_logs

def populate_month_info(shift_schedule, call_centre_scenario):
    call_centre_scenario_operation = pd.DataFrame()
    cost_and_abandonments = pd.DataFrame()
    unique_days = call_centre_scenario['Day'].unique()
    initial_agent_logs = generate_agent_logs(shift_schedule)
    start = datetime.now()
    
    for day in unique_days:
        day_scenario = call_centre_scenario[call_centre_scenario['Day'] == day]
        agent_logs = initial_agent_logs.copy()
        
        answered_calls = 0
        abandoned_calls = 0
        day_cost = calculate_cost(shift_schedule)
        
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
                        answered_calls = answered_calls + 1
                        break
                    
                if call_features.shape[1] != 6:
                    call_features = call_features.assign(Waiting_time = call_features.iloc[0,2])
                    call_features = call_features.assign(Call_answered = 'no')
                    call_centre_scenario_operation = call_centre_scenario_operation.append(call_features)
                    abandoned_calls = abandoned_calls + 1
                
            else:
                call_features = call_features.assign(Waiting_time = call_features.iloc[0,2])
                call_features = call_features.assign(Call_answered = 'no')
                call_centre_scenario_operation = call_centre_scenario_operation.append(call_features)
                abandoned_calls = abandoned_calls + 1
        
        append_day_features = np.zeros((1,4))
        
        append_day_features[0,0] = day
        append_day_features[0,1] = answered_calls
        append_day_features[0,2] = abandoned_calls
        append_day_features[0,3] = day_cost
        
        cost_and_abandonments = cost_and_abandonments.append(pd.DataFrame(append_day_features))
        print('Day ',day , 'appended at: ', datetime.now() - start)
        
    return cost_and_abandonments

'''FOR CONSTISTENCY CONVERT SCENARIO ATTRIBUTES IN TO HOURS INSTEAD OF CONVERTING SHIFT INFO INTO SECONDS'''

shift_info = pd.read_csv('../Part 1 datasets/Initial results/Results_4_64_1.0_sample7.csv')
shift_info = shift_info[["Start_time","End_time","Break_start","Break_end","Staff_number"]]
shift_info[["Start_time","End_time","Break_start","Break_end"]] = shift_info[["Start_time","End_time","Break_start","Break_end"]]*60*60
shift_info = shift_info[shift_info["Staff_number"] > 0]
shift_info.Staff_number = shift_info.Staff_number.astype(int)
shift_info = shift_info.reset_index(drop = True)

call_info = pd.read_csv('../Part 1 datasets/Scenario_4_64_1.0_4.csv')
call_info = call_info[call_info.Day.isin(range(1,15))]

cost_per_day = populate_month_info(shift_info, call_info)
cost_per_day.columns = ['Day', 'Answered_calls', 'Abandoned_calls', 'Day_cost']
cost_per_day = cost_per_day.reset_index(drop = True)

file_name = 'Costs_and_abandonments_' + 'scenario_4_64_1.0_4'
export_data(cost_per_day, file_name)

