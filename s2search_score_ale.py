from s2search.rank import S2Ranker
import time
import os
import os.path as path
import sys
import yaml
import math
import numpy as np
import pandas as pd
import json
from getting_data import load_sample, read_conf
from s2search_score_pipelining import get_scores

model_dir = './s2search_data'
data_dir = str(path.join(os.getcwd(), 'pipelining'))
ranker = None
def_quantile_config = {'title': 5, 'abstract': 10, 'venue': 20, 'authors': 10, 'year': 5, 'n_citations': 0.1}

def init_ranker():
    global ranker
    if ranker == None:
        print(f'Loading ranker model...')
        st = time.time()
        ranker = S2Ranker(model_dir)
        et = round(time.time() - st, 2)
        print(f'Load the s2 ranker within {et} sec')


# def get_scores(query, paper_list):
#     init_ranker()
#     scores = []
#     if len(paper_list) > 1000:
#         curr_idx = 0
#         while curr_idx < len(paper_list):
#             end_idx = curr_idx + 1000 if curr_idx + 1000 < len(paper_list) else len(paper_list)
#             curr_list = paper_list[curr_idx: end_idx]
#             scores.extend(ranker.score(query, curr_list))
#             curr_idx += 1000
#     else:
#         scores = ranker.score(query, paper_list)
#     return scores

def save_pdp_to_npz(exp_dir_path, npz_file_path, quantile, ale_result, values_for_rug):
    scores_dir = path.join(exp_dir_path, 'scores')
    if not path.exists(str(scores_dir)):
        os.mkdir(str(scores_dir))
    print(f'\tsave ale data to {npz_file_path}')
    if values_for_rug == None:
        np.savez_compressed(npz_file_path, quantile=quantile, ale_result=ale_result)
    else:
        np.savez_compressed(npz_file_path, quantile=quantile, ale_result=ale_result, values_for_rug=values_for_rug)
        
def save_pdp_to_npz_2w(exp_dir_path, npz_file_path, quantile1, quantile2, ale_result):
    scores_dir = path.join(exp_dir_path, 'scores')
    if not path.exists(str(scores_dir)):
        os.mkdir(str(scores_dir))
    print(f'\tsave ale data to {npz_file_path}')
    np.savez_compressed(npz_file_path, quantile_1=quantile1, quantile_2=quantile2, ale_result=ale_result)

def divide_by_percentile(df, sorted_column_name, use_interval_not_quantile, quantile_interval, just_interval, with_nighbor=True):
    data_len = len(df.index)
    if use_interval_not_quantile:
        actual_interval = just_interval
    else:
        actual_interval = math.ceil(data_len * (quantile_interval * 0.01))
    grids = []
    quantiles = []
    curr_idx = 0
    while curr_idx < data_len:
        upper_idx = curr_idx + actual_interval - 1 if curr_idx + actual_interval - 1 < data_len else data_len - 1
        grid = {
            'lower_z': df[sorted_column_name].iloc[curr_idx],
            'upper_z': df[sorted_column_name].iloc[upper_idx],
        }
        if with_nighbor:
            grid['neighbor']= df.iloc[curr_idx: upper_idx + 1]

        grids.append(grid)
            
        quantiles.append(df[sorted_column_name].iloc[upper_idx])
        curr_idx += actual_interval
    # print(use_interval_not_quantile, actual_interval, len(quantiles))

    return grids, quantiles

def get_ale(grids, feature_key, query, centered = False):
    curr_accumulated = 0
    ale_result_list = []
    for grid in grids:
        lv = grid['lower_z']
        uv = grid['upper_z']
        neighbor = grid['neighbor']
        
        points_ul_pairs = []
        for idx, row in neighbor.iterrows():
            upper_variant = {**row}
            upper_variant[feature_key] = uv
            
            lower_variant = {**row}
            lower_variant[feature_key] = lv
            points_ul_pairs.append([upper_variant, lower_variant])
        
        all_diff_in_this = fn_over_grid_2(query, points_ul_pairs)
        curr_accumulated += np.mean(all_diff_in_this)
        ale_result_list.append(curr_accumulated)
    
    if centered: 
        mean = np.mean(ale_result_list)
        return [(x - mean) for x in ale_result_list]
    else:
        return ale_result_list

def fn_over_grid_2(query, points_ul_pairs):
    diff_list = []
    paper_list = []
    for point_pair in points_ul_pairs:
        upper, lower = point_pair
        paper_list.append(upper)
        paper_list.append(lower)
    
    scores = get_scores(query, paper_list)
        # scores = list(map(lambda x: get_scores(query, [x]), paper_list))

    idx = 0
    while idx < len(scores):
        upper_score = scores[idx]
        lower_score = scores[idx + 1]
        diff = upper_score - lower_score
        if diff > 100:
            # print(diff, idx, int(idx / 2))
            # print(points_ul_pairs[int(idx / 2)])
            diff_in_two = get_scores(query, points_ul_pairs[int(idx / 2)])
            # print(diff_in_two)
            diff_list.append(diff_in_two[1] - diff_in_two[0])
            print(f'abnormal scores occur {diff}, adjust to {diff_in_two[1] - diff_in_two[0]}')
        else:
            diff_list.append(diff)
        idx += 2
    
    return diff_list

def get_bin_size(interval_config, quantile_config, feature_name):
    just_interval = None
    quantile_interval = None
    if interval_config == None or interval_config.get(feature_name) == None:
        use_interval_not_quantile = False
        quantile_interval = quantile_config.get(feature_name) if quantile_config != None and quantile_config.get(feature_name) != None else def_quantile_config.get(feature_name) 
    else:
        use_interval_not_quantile = True
        just_interval = interval_config.get(feature_name)

    return just_interval, quantile_interval, use_interval_not_quantile

def split_df_by_cluster(df):
    sorted_scores = list(df['score'])
    cluster_c_end_idx = 0
    cluster_b_end_idx = 0
    for i in range(len(sorted_scores)):
        score = sorted_scores[i]
        if score < -10:
            cluster_c_end_idx = i
        elif score < 0:
            cluster_b_end_idx = i
            
    # print(cluster_c_end_idx, cluster_b_end_idx, cluster_a_end_idx)
    return df.iloc[0 : cluster_c_end_idx + 1], df.iloc[cluster_c_end_idx + 1: cluster_b_end_idx + 1], df.iloc[cluster_b_end_idx + 1 : len(sorted_scores)]  

def get_grids_and_quantiles(f1_df, f1_feature_name, interval_config, quantile_config):
    f1_itv, f1_quant, f1_use_itv = get_bin_size(interval_config, quantile_config, f1_feature_name)
    
    if f1_feature_name == 'year' or f1_feature_name == 'n_citations':
        f1_grids, f1_quantiles = divide_by_percentile(f1_df, f1_feature_name, f1_use_itv, f1_quant, f1_itv)
    else:
        f1_clst_c, f1_clst_b, f1_clst_a = split_df_by_cluster(f1_df)

        f1_c_grids, f1_c_quantiles = divide_by_percentile(f1_clst_c, f1_feature_name, f1_use_itv, 
                                                f1_quant, f1_itv)
        f1_b_grids, f1_b_quantiles = divide_by_percentile(f1_clst_b, f1_feature_name, f1_use_itv, 
                                                f1_quant, f1_itv)
        f1_a_grids, f1_a_quantiles = divide_by_percentile(f1_clst_a, f1_feature_name, f1_use_itv, 
                                                f1_quant, f1_itv)
        f1_grids = []
        f1_quantiles = []

        f1_grids.extend(f1_c_grids)
        f1_grids.extend(f1_b_grids)
        f1_grids.extend(f1_a_grids)
        f1_quantiles.extend(f1_c_quantiles)
        f1_quantiles.extend(f1_b_quantiles)
        f1_quantiles.extend(f1_a_quantiles)
    
    return f1_grids, f1_quantiles

def find_mutial_neighbor(n1, n2):
    # m = pd.DataFrame(columns=n1.columns)
    # for idx1, row1 in n1.iterrows():
    #     for idx2, row2 in n2.iterrows():
    #         if row1['id'] == row2['id']:
    #             m = pd.concat([m, pd.DataFrame(data=[row1], columns = n1.columns)], ignore_index=True)
    # return m
    m = pd.DataFrame(columns=n1.columns)
    id_in_n1 = {}
    # st = time.time()
    for idx1, row1 in n1.iterrows():
        id_in_n1[row1['id']] = ''
    
    for idx2, row2 in n2.iterrows():
      if id_in_n1.get(row2['id']) != None:
        m = pd.concat([m, pd.DataFrame(data=[row1], columns = n1.columns)], ignore_index=True)
      
    # print(f'find mutual n within {round(time.time() - st)} sec')
    return m

def compute_and_save(output_exp_dir, output_data_sample_name, query, quantile_config, interval_config, data_exp_name, data_sample_name):
    print(output_exp_dir, output_data_sample_name, query, data_exp_name, data_sample_name)
    categorical_features = [
        'title', 
        'abstract',
        'venue',
        'authors',
        'year',
        'n_citations',
    ]
    for feature_name in categorical_features:
        npz_file_path = path.join(output_exp_dir, 'scores',
                                  f"{output_data_sample_name}_1w_ale_{feature_name}.npz")
        if not os.path.exists(npz_file_path):
            df = load_sample(data_exp_name, data_sample_name, sort=feature_name, rank_f=get_scores, query=query)
            values_for_rug = df[feature_name].to_list() if feature_name == 'year' or feature_name == 'n_citations' else None
                
            st = time.time()
            just_interval, quantile_interval, use_interval_not_quantile = get_bin_size(interval_config, quantile_config, feature_name)

            grids, quantiles = divide_by_percentile(df, feature_name, use_interval_not_quantile, quantile_interval, just_interval)
            ale_result = get_ale(grids, feature_name, query)
            
            et = round(time.time() - st, 6)
            print(f'\tcompute ale data for {output_data_sample_name}_1w_ale_{feature_name} within {et} sec')
            if (feature_name == 'authors'):
                quantiles = [str(x) for x in quantiles]
            save_pdp_to_npz(output_exp_dir, npz_file_path, quantiles, ale_result, values_for_rug)
        else:
            print(f'\t{npz_file_path} exist, should skip')
            
    for i in range(len(categorical_features)):
        f1_feature_name = categorical_features[i]
        for j in range(i + 1, len(categorical_features)):
            f2_feature_name = categorical_features[j]
            npz_file_path = path.join(output_exp_dir, 'scores',
                                  f"{output_data_sample_name}_2w_ale_{f1_feature_name}_{f2_feature_name}.npz")
            if not os.path.exists(npz_file_path):
                print(f1_feature_name, f2_feature_name)
                st = time.time()
                f1_df = load_sample(data_exp_name, data_sample_name, sort=f1_feature_name, del_f = ['s2_id'], rank_f=get_scores, query=query)
                f2_df = load_sample(data_exp_name, data_sample_name, sort=f2_feature_name, del_f = ['s2_id'], rank_f=get_scores, query=query)
                
                f1_grids, f1_quantiles = get_grids_and_quantiles(f1_df, f1_feature_name, interval_config, quantile_config)
                f2_grids, f2_quantiles = get_grids_and_quantiles(f2_df, f2_feature_name, interval_config, quantile_config)
                
                if (f1_feature_name == 'authors'):
                    f1_quantiles = [str(x) for x in f1_quantiles]
                if (f2_feature_name == 'authors'):
                    f2_quantiles = [str(x) for x in f2_quantiles]
                
                ale_values = np.zeros([len(f2_grids), len(f1_grids)])
                
                for k in range(len(f1_grids)):
                    f1_grid = f1_grids[k]
                    f1_upper =  f1_grid['upper_z']
                    f1_lower =  f1_grid['lower_z']
                    f1_neighbor =  f1_grid['neighbor']
                    for l in range(len(f2_grids)):
                        f2_grid = f2_grids[l]
                        f2_upper = f2_grid['upper_z']
                        f2_lower = f2_grid['lower_z']
                        f2_neighbor =  f1_grid['neighbor']
                        
                        mutual_neighbor = find_mutial_neighbor(f1_neighbor, f2_neighbor)
                        
                        all_diff_in_neignborhood = []
                        
                        four_corner_paper = []
                        for idx, row in mutual_neighbor.iterrows():
                            a = {**row}
                            a[f1_feature_name] = f1_lower
                            a[f2_feature_name] = f2_lower
                            
                            b = {**row}
                            b[f1_feature_name] = f1_upper
                            b[f2_feature_name] = f2_lower
                            
                            c = {**row}
                            c[f1_feature_name] = f1_lower
                            c[f2_feature_name] = f2_upper
                            
                            d = {**row}
                            d[f1_feature_name] = f1_upper
                            d[f2_feature_name] = f2_upper
                            
                            # a_s, b_s, c_s, d_s = get_scores(query, [a, b, c, d])
                            # a_s, b_s, c_s, d_s = [0,0,0,0]
                            four_corner_paper.extend([a, b, c, d])
                            # diff = (d_s - c_s) - (b_s - a_s)
                            # all_diff_in_neignborhood.append(diff)
                        
                        four_corner_paper_scores = get_scores(query, four_corner_paper)
                        idx = 0
                        while idx < len(four_corner_paper_scores):
                            a_s, b_s, c_s, d_s = four_corner_paper_scores[idx: idx + 4]
                            diff = round((d_s - c_s) - (b_s - a_s), 14)
                            # if a_s > 100 or b_s > 100 or c_s > 100 or d_s > 100:
                            #     print(diff, idx, int(idx / 4), '|', a_s, b_s, c_s, d_s)
                            #     a_s, b_s, c_s, d_s = [get_scores(query, [x])[0] for x in four_corner_paper[idx: idx + 4]]
                            #     # print('|-', a_s, b_s, c_s, d_s)
                            #     all_diff_in_neignborhood.append(round((d_s - c_s) - (b_s - a_s), 14))
                            #     pass
                            # else:
                            all_diff_in_neignborhood.append(diff)
                            idx += 4
                        
                        accumulated_f1_value = 0 if k == 0 else ale_values[l][k - 1]
                        accumulated_f2_value = 0 if l == 0 else ale_values[l - 1][k]
                        # ale_values[l][k] = accumulated_f1_value + accumulated_f2_value + np.mean(all_diff_in_neignborhood)
                        accumulated_value = 0
                        # accumulated_value = 0 if k == 0 and l == 0 else ale_values[l - 1][k - 1]
                        ale_values[l][k] = accumulated_f1_value + accumulated_f2_value + accumulated_value + np.mean(all_diff_in_neignborhood)

                et = round(time.time() - st, 6)
                print(f'\tcompute ale data for {output_data_sample_name}_2w_ale_{f1_feature_name}_{f2_feature_name} within {et} sec')
                save_pdp_to_npz_2w(output_exp_dir, npz_file_path, f1_quantiles, f2_quantiles, ale_values)

         
def get_ale_and_save_score(exp_dir_path, exp_name):
    des, sample_configs, sample_from_other_exp = read_conf(exp_dir_path)
    
    tested_sample_list = []
    
    for sample_name in sample_configs:
        if sample_name in sample_from_other_exp.keys():
            other_exp_name, data_file_name = sample_from_other_exp.get(sample_name)
            tested_sample_list.append({'exp_name': other_exp_name, 'data_sample_name': sample_name, 'data_source_name': data_file_name.replace('.data', '')})
        else:
            tested_sample_list.append({'exp_name': exp_name, 'data_sample_name': sample_name, 'data_source_name': sample_name})

    for tested_sample_config in tested_sample_list:
        
        tested_sample_name = tested_sample_config['data_sample_name']
        tested_sample_data_source_name = tested_sample_config['data_source_name']
        tested_sample_from_exp = tested_sample_config['exp_name']
        
        task = sample_configs.get(tested_sample_name)
        if task != None:
            print(f'computing ale for {tested_sample_name}')
            for t in task:
                query = t['query']
                quantile_config = t.get('quantiles')
                interval_config = t.get('intervals')
                compute_and_save(
                    exp_dir_path, tested_sample_name, query, quantile_config, interval_config,
                    tested_sample_from_exp, tested_sample_data_source_name)
        else:
            print(f'**no config for tested sample {tested_sample_name}')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        exp_list = sys.argv[1:]
        for exp_name in exp_list:
            exp_dir_path = path.join(data_dir, exp_name)
            if path.isdir(exp_dir_path):
                get_ale_and_save_score(exp_dir_path, exp_name)
            else:
                print(f'**no exp dir {exp_dir_path}')
