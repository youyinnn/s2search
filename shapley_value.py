import numpy as np, sys, os
from getting_data import get
from feature_masking import masking_options
from math import factorial, ceil
import pandas as pd
import multiprocessing
from multiprocessing import Pool

feature_key_list = ['t', 'abs', 'v', 'au', 'y', 'c']
all_ = 'tabsvauyc'

# number of all the player
p = len(feature_key_list)
f_p = factorial(p)

# get binary code string of a coalition
def get_code(coalition):
    codes = []
    for k in feature_key_list:
        if k in coalition:
            codes.append('1')
        else:
            codes.append('0')
    return ''.join(codes)

# get coalition string of a binary code
def get_coalition_by_code(codes):
    coalition = []
    for i in range(len(codes)):
        code = codes[i]
        if (code == '1'):
            coalition.append(feature_key_list[i])
    return ''.join(coalition)

def reverse_code(code):
    return ['1' if c == '0' else '0' for c in list(code)]
            
# print(combination_list[31], get_code(combination_list[31]), get_coalition_by_code(get_code(combination_list[31])))

def get_ith_instance_pred_by_coalition(i, coalition, sample_data_and_config_arr):
    if coalition == '+':
        return -17.16172365
    elif coalition == all_:
        return sample_data_and_config_arr[0]['origin'][i]
    for task in sample_data_and_config_arr:
        masking_option_keys = task['masking_option_keys']
        for j in range(len(masking_option_keys)):
            co = masking_option_keys[j]
            if co == coalition:
                return task['feature_stack'][j][i]
            
# print(get_ith_instance_pred_by_coalition(-1, 't'))

# -17.16172365
# get_scores('Machine Learning', [{}])

def get_individual_sv_list(task_args):
    start_idx, end_idx, exclusive_combination_list, sample_data_and_config_arr = task_args
    # print(f'worker {os.getpid()} for {start_idx} to { end_idx}')
    rs = []
    for i in range(start_idx, end_idx):
        sv_for_all_feature = []
        for player in exclusive_combination_list.keys():
            coalition_and_pairs = exclusive_combination_list[player]
            sum_ = 0
            for pair in coalition_and_pairs:
                coalition_with_player, coalition_without_player = pair
                code_for_coalition_without_player = get_code(coalition_without_player)
                elem_number_of_s = list(code_for_coalition_without_player).count('1')
                f_s = factorial(elem_number_of_s)
                # print(coalition_with_player, coalition_without_player, code_for_coalition_without_player, f_s )
                score_of_coalition_with_player = get_ith_instance_pred_by_coalition(i, coalition_with_player, sample_data_and_config_arr)
                score_of_coalition_without_player = get_ith_instance_pred_by_coalition(i, coalition_without_player, sample_data_and_config_arr)
                
                # print(coalition_with_player, coalition_without_player, code_for_coalition_without_player, 
                #       elem_number_of_s, score_of_coalition_with_player, score_of_coalition_without_player, (score_of_coalition_with_player - score_of_coalition_without_player))
                
                sum_ += f_s * factorial((p - 1 - elem_number_of_s)) * (score_of_coalition_with_player - score_of_coalition_without_player)
            sv_for_all_feature.append(sum_ / f_p)
        
        # instane_sv.loc[len(instane_sv.index)] = sv_for_all_feature
        rs.append(sv_for_all_feature)
    
    return rs

def get_shapley_value(exp_name, sample_name):
    sample_data_and_config_arr = get(exp_name, sample_name)

    combination_list = ['+']

    for task in sample_data_and_config_arr:
        masking_option_keys = task['masking_option_keys']
        for i in range(len(masking_option_keys)):
            co = masking_option_keys[i]
            code_of_co = get_code(co)
            r_code = reverse_code(code_of_co)
            masking_option_keys[i] = get_coalition_by_code(r_code)
            # print(co, get_coalition_by_code(r_code))
        combination_list.extend([co for co in masking_option_keys])

    exclusive_combination_list = {}

    for feature_name in feature_key_list:
        exclusive_combination_list[feature_name] = \
            [coalition for coalition in combination_list if feature_name not in coalition]
            
    for player in exclusive_combination_list.keys():
        coalitions = exclusive_combination_list[player]
        for i in range(len(coalitions)):
            coalition_without_player = coalitions[i]
            if coalition_without_player == '+':
                coalitions[i] = [player, '+']
            else:
                codes = get_code(coalition_without_player)
                player_idx = feature_key_list.index(player)
                codes = codes[:player_idx] + '1' + codes[player_idx + 1:]
                coalition_with_player = get_coalition_by_code(codes)
                coalitions[i] = [coalition_with_player, coalition_without_player]

    data_len = sample_data_and_config_arr[0]['origin'].shape[0]

    work_load = int(multiprocessing.cpu_count()) - 2
    paper_limit_for_a_worker = ceil(data_len / work_load)

    task_args = []

    curr_idx = 0
    while curr_idx < data_len:
        end_idx = curr_idx + paper_limit_for_a_worker if curr_idx + paper_limit_for_a_worker < data_len else data_len
        task_args.append([
            curr_idx, 
            end_idx, 
            exclusive_combination_list, 
            sample_data_and_config_arr
        ])
        curr_idx += paper_limit_for_a_worker

    with Pool(processes=work_load) as worker:
        rs = worker.map_async(get_individual_sv_list, task_args)
        individual_sv = rs.get()
        
    data = []
        
    for p in individual_sv:
        for o in p:
            data.append(o)
    
    return pd.DataFrame(columns=[f'{f}_sv' for f in feature_key_list], data=data)