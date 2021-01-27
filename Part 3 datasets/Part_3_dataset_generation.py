import pandas as pd
import numpy as np
import math
import random
import csv


def export_data(dataset, filename):
    file_name = filename + '.csv'
    dataset.to_csv(file_name, index=False)


def calculate_rates(application_info, agent_performance_info):
    
    calls_abandoned = sum(application_info["CallsAbandoned"])
    calls_abandoned_delay = sum(application_info["CallsAbandonedDelay"])
    abandon_rate = calls_abandoned / calls_abandoned_delay
    
    calls_answered = sum(agent_performance_info["CallsAnswered"])
    talk_time = sum(agent_performance_info["TalkTime"])
    service_rate = calls_answered / talk_time
    
    return abandon_rate, service_rate


def generate_interval_data(calls, call_interval, day, service_rate, abandonment_rate):
    
    interval_data = pd.DataFrame()
    
    for i in range(calls):
        p2 = random.random()
        p3 = random.random()
        append_call_features = np.zeros((1,4))
    
        arrival_time = random.randint(call_interval, call_interval + 15 * 60 - 1)
        service_time = round(-1 * np.log(1-p2) / service_rate)
        patience_time = round(-1 * np.log(1-p3) / abandonment_rate)
        
        append_call_features[0,0] = arrival_time
        append_call_features[0,1] = service_time
        append_call_features[0,2] = patience_time
        append_call_features[0,3] = day
        
        interval_data = interval_data.append(pd.DataFrame(append_call_features))
    
    return interval_data


def generate_days(call_data, abandon_rate, service_rate):
    
    scenario_data = pd.DataFrame()
    rows = call_data.shape[0]

    for i in range(rows):
        calls = call_data.iloc[i,0]
        day = call_data.iloc[i,1]
        call_interval = call_data.iloc[i,2]
        
        interval_data = generate_interval_data(calls, call_interval, day, service_rate, abandon_rate)
        scenario_data = scenario_data.append(pd.DataFrame(interval_data))
    
    return scenario_data


application_data = pd.read_csv('apps_primary_refined_processed.csv')
agent_performance_data = pd.read_csv('agent_perf_refined_processed.csv')
ptd_agents = pd.read_csv('agents_ptd_refined.csv')

application_data = application_data[application_data["CallMonth"] == "April"]
application_data = application_data[application_data["CallInterval"] >= 6*60*60]
application_data = application_data[application_data["CallInterval"] < 23*60*60]
application_data["CallsAbandoned"] = application_data["CallsOffered"] - application_data["CallsAnswered"]
application_data = application_data[application_data["CallsAbandoned"] > 0]
application_data = application_data.reset_index(drop = True)

agent_performance_data = agent_performance_data[agent_performance_data["CallMonth"] == "April"]
agent_performance_data = agent_performance_data[agent_performance_data["CallInterval"] >= 6*60*60]
agent_performance_data = agent_performance_data[agent_performance_data["CallInterval"] < 23*60*60]
agent_performance_data = agent_performance_data[agent_performance_data["CallsAnswered"] > 0]
# remove ptd agents
agent_performance_data = agent_performance_data[~agent_performance_data.TelsetLoginID.isin(ptd_agents.AgentID)]
agent_performance_data = agent_performance_data.reset_index(drop = True)

abandon_rate, service_rate = calculate_rates(application_data, agent_performance_data)

application_data_slice = application_data[["CallsAbandoned","CallDay","CallInterval"]]
agent_performance_data_slice = agent_performance_data[["CallsAnswered","CallDay","CallInterval"]]

abandoned_calls_data = generate_days(application_data_slice, abandon_rate, service_rate)
answered_calls_data = generate_days(agent_performance_data_slice, abandon_rate, service_rate)

consolidated_data = abandoned_calls_data.append(answered_calls_data)
consolidated_data.columns = ['Arrival_time', 'Service_duration', 'Time_to_abandon', 'Day']
consolidated_data = consolidated_data.sort_values(by = ['Day', 'Arrival_time'])
consolidated_data = consolidated_data.reset_index(drop = True)

export_data(consolidated_data, 'April_processed_calls_using_offered_calls')
export_data(agent_performance_data, 'April_processed_agent_performance_using_offered_calls')

'''
for i in range(rows):
    scenario_data = generate_day(variable_combo_filtered.iloc[i,0], variable_combo_filtered.iloc[i,1], variable_combo_filtered.iloc[i,2], variable_combo_filtered.iloc[i,3])
    scenario_text = str(variable_combo_filtered.iloc[i,0]) + '_' + str(variable_combo_filtered.iloc[i,1]) + '_' + str(variable_combo_filtered.iloc[i,2]) + '_' + str(variable_combo_filtered.iloc[i,3])
    #scenario_text = scenario_text.replace(".","")
    scenario_data.columns = ['Arrival_time', 'Service_duration', 'Time_to_abandon', 'Day']
    export_data(scenario_data, scenario_text)

#consolidate(1)
consolidate(2)
#consolidate(4)
'''