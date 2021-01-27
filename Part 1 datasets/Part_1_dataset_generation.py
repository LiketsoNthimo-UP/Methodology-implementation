import pandas as pd
import numpy as np
import math
import random
import csv

def generate_call_features(arrival_rate, service_rate, abandonment_rate):
    arrival_rate = arrival_rate/60/60
    service_rate = service_rate/60/60
    abandonment_rate = abandonment_rate/60/60

    p1 = random.random()
    p2 = random.random()
    p3 = random.random()

    inter_arrival_time = -1 * np.log(1-p1) / arrival_rate
    service_time = -1 * np.log(1-p2) / service_rate

    if abandonment_rate == 0:
        patience_time = 12 * 60 * 60
    else:
        patience_time = -1 * np.log(1-p3) / abandonment_rate
    
    return (inter_arrival_time, service_time, patience_time)

def find_current_arrival_rate(average_arrival_rate, RA, current_time):
    current_time = current_time / 60 / 60
    return average_arrival_rate * (1 + RA * math.sin(math.pi * current_time / 4))

def export_data(dataset, scenario_combo):
    file_name = 'Scenario_' + scenario_combo + '.csv'
    dataset.to_csv(file_name, index=False)

def generate_day(service_rate, offered_load, RA, abandonment_rate):
    average_arrival_rate = service_rate * offered_load
    current_time = 0
    current_arrival_rate = find_current_arrival_rate(average_arrival_rate, RA, current_time)
    scenario_data = pd.DataFrame()

    for day in range(1,11):
        call_features = generate_call_features(current_arrival_rate, service_rate, abandonment_rate)
        arrival_time = current_time + call_features[0]
        append_call_features = np.zeros((1,4))

        while arrival_time < 12 * 60 * 60:
            append_call_features[0,0] = round(arrival_time)
            append_call_features[0,1] = round(call_features[1])
            append_call_features[0,2] = round(call_features[2])
            append_call_features[0,3] = day
            scenario_data = scenario_data.append(pd.DataFrame(append_call_features))
            
            current_time = current_time + call_features[0]
            current_arrival_rate = find_current_arrival_rate(average_arrival_rate, RA, current_time)
            call_features = generate_call_features(current_arrival_rate, service_rate, abandonment_rate)
            arrival_time = current_time + call_features[0]
        
        current_time = 0
        current_arrival_rate = find_current_arrival_rate(average_arrival_rate, RA, current_time)
        arrival_time = 0
    
    return scenario_data

def consolidate(selection):
    variable_combo = pd.read_csv('Variable_combo.csv')
    variable_combo_filtered = variable_combo[variable_combo['Service_rate'] == selection]
    variable_combo_filtered = variable_combo_filtered[variable_combo_filtered['Offered_load'] == 64]
    variable_combo_filtered = variable_combo_filtered[variable_combo_filtered['Relative_amplitude'] == 1]
    variable_combo_filtered = variable_combo_filtered[variable_combo_filtered['Abandonment_rate'] == 2]
    
    rows = variable_combo_filtered.shape[0]
    
    for i in range(rows):
        scenario_data = generate_day(variable_combo_filtered.iloc[i,0], variable_combo_filtered.iloc[i,1], variable_combo_filtered.iloc[i,2], variable_combo_filtered.iloc[i,3])
        scenario_text = str(variable_combo_filtered.iloc[i,0]) + '_' + str(variable_combo_filtered.iloc[i,1]) + '_' + str(variable_combo_filtered.iloc[i,2]) + '_' + str(variable_combo_filtered.iloc[i,3])
        #scenario_text = scenario_text.replace(".","")
        scenario_data.columns = ['Arrival_time', 'Service_duration', 'Time_to_abandon', 'Day']
        export_data(scenario_data, scenario_text)

#consolidate(1)
consolidate(2)
#consolidate(4)