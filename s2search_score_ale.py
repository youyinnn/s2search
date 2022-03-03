import time
import os
import os.path as path
import sys
import math
import multiprocessing
import numpy as np
import pandas as pd
from getting_data import load_sample, read_conf
from s2search_score_pipelining import get_scores

from sys import platform
from multiprocessing import Pool

data_dir = str(path.join(os.getcwd(), 'pipelining'))
def_quantile_config = {'title': 5, 'abstract': 10, 'venue': 20, 'authors': 10, 'year': 5, 'n_citations': 0.1}
p_number = int(multiprocessing.cpu_count() - (2 if platform == "darwin" else 0))

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

    return grids, quantiles

def get_ale(grids, feature_key, query, centered = False):
    curr_accumulated = 0
    ale_result_list = []
    points_ul_pairs_2 = []
    gaps = []
    for grid in grids:
        lv = grid['lower_z']
        uv = grid['upper_z']
        neighbor = grid['neighbor']
        
        for idx, row in neighbor.iterrows():
            upper_variant = {**row}
            upper_variant[feature_key] = uv
            
            lower_variant = {**row}
            lower_variant[feature_key] = lv
            points_ul_pairs_2.append([upper_variant, lower_variant])
        
        gaps.append(len(neighbor.index))
    
    all_diff = fn_over_grid_2(query, points_ul_pairs_2)
    all_diffs_split = []
    idx = 0
    for gap in gaps:
        all_diffs_split.append(all_diff[idx: idx + gap])
        idx += gap
    
    for diffs in all_diffs_split:
        curr_accumulated += np.mean(diffs)
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
    
    scores = get_scores(query, paper_list, task_name='ale-1w', ptf=False)
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
    f1_grids, f1_quantiles = divide_by_percentile(f1_df, f1_feature_name, f1_use_itv, f1_quant, f1_itv)
    return f1_grids, f1_quantiles

def find_mutual_neighbor(n1, n2):
    m = pd.DataFrame(columns=n1.columns, data=[])
    merge_id = list(pd.merge(n1['id'], n2['id'], on=['id'], how='inner')['id'])

    for idx2, row2 in n2.iterrows():
      if row2['id'] in merge_id:
        m = pd.concat([m, pd.DataFrame(data=[row2], columns = n1.columns)], ignore_index=True)

    return m

def get_four_corner_papers(args):
    four_corner_paper = []
    f1_feature_name, f1_lower, f1_upper,f1_neighbor, f2_feature_name, f2_lower,f2_upper,f2_neighbor, row, col, ale_values_shape = args
    mutual_neighbor = find_mutual_neighbor(f1_neighbor, f2_neighbor)
    for idx, row_data in mutual_neighbor.iterrows():
        a = {**row_data}
        a[f1_feature_name] = f1_lower
        a[f2_feature_name] = f2_lower
        
        b = {**row_data}
        b[f1_feature_name] = f1_upper
        b[f2_feature_name] = f2_lower
        
        c = {**row_data}
        c[f1_feature_name] = f1_lower
        c[f2_feature_name] = f2_upper
        
        d = {**row_data}
        d[f1_feature_name] = f1_upper
        d[f2_feature_name] = f2_upper
        four_corner_paper.extend([a, b, c, d])
        
    print("\r", f'finding mutual neignbor: ({row}, {col}), {ale_values_shape}', end="" , flush=True)
    return {
        'four_corner_paper': four_corner_paper,
        'row': row,
        'col': col
    }

def compute_and_save(output_exp_dir, output_data_sample_name, query, quantile_config, interval_config, data_exp_name, data_sample_name, for_2way):
    print(output_exp_dir, output_data_sample_name, query, data_exp_name, data_sample_name, f'2-way {for_2way}')
    categorical_features = [
        'title', 
        'abstract',
        'venue',
        'authors',
        'year',
        'n_citations',
    ]
    if not for_2way:
        for feature_name in categorical_features:
            npz_file_path = path.join(output_exp_dir, 'scores',
                                    f"{output_data_sample_name}_1w_ale_{feature_name}.npz")
            if not os.path.exists(npz_file_path):
                df = load_sample(data_exp_name, data_sample_name, sort=feature_name, rank_f=get_scores, query=query)
                values_for_rug = df[feature_name].to_list() if feature_name == 'year' or feature_name == 'n_citations' else None
                    
                st = time.time()
                # grids, quantiles = divide_by_percentile(df, feature_name, use_interval_not_quantile, quantile_interval, just_interval)
                grids, quantiles = get_grids_and_quantiles(df, feature_name, interval_config, quantile_config)
                ale_result = get_ale(grids, feature_name, query)

                et = round(time.time() - st, 6)
                print(f'\tcompute ale data for {output_data_sample_name}_1w_ale_{feature_name} within {et} sec')
                if (feature_name == 'authors'):
                    quantiles = [str(x) for x in quantiles]
                save_pdp_to_npz(output_exp_dir, npz_file_path, quantiles, ale_result, values_for_rug)
            else:
                print(f'\t{npz_file_path} exist, should skip')
    
    else:      
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
                    
                    # row: f2_len col: f1_len
                    ale_values                  = np.zeros([len(f2_grids), len(f1_grids)])
                    neighbors_number_per_grids  = np.zeros([len(f2_grids), len(f1_grids)])
                    
                    four_corner_papers = []

                    # f1_grids_len * f2_grids_len * (4 * number_of_neighbors)
                    task_args = []
                    
                    for row in range(len(f2_grids)):
                        f2_grid = f2_grids[row]
                        f2_upper =  f2_grid['upper_z']
                        f2_lower =  f2_grid['lower_z']
                        f2_neighbor =  f2_grid['neighbor']
                        for col in range(len(f1_grids)):
                            f1_grid = f1_grids[col]
                            f1_upper = f1_grid['upper_z']
                            f1_lower = f1_grid['lower_z']
                            f1_neighbor =  f1_grid['neighbor']
                            
                            task_args.append([
                                f1_feature_name,f1_lower,f1_upper,f1_neighbor,
                                f2_feature_name,f2_lower,f2_upper,f2_neighbor,
                                row, col,
                                ale_values.shape
                            ])
                            
                    with Pool(processes=p_number) as worker:
                        rs = worker.map_async(get_four_corner_papers, task_args)
                        task_rs = rs.get()

                    print(' - ')        
                            
                    for rs in task_rs:
                        fcp = rs['four_corner_paper']
                        four_corner_papers.extend(fcp)
                        neighbors_number_per_grids[rs['row']][rs['col']] = len(fcp)
                    
                    
                    four_corner_paper_scores = get_scores(query, four_corner_papers)

                    curr_idx = 0
                    for row in range(len(f2_grids)):
                        for col in range(len(f1_grids)):
                            number_of_neignbor_with_four_corners = int(neighbors_number_per_grids[row][col])            
                            neignbor_with_four_corner_scores = four_corner_paper_scores[curr_idx : curr_idx + number_of_neignbor_with_four_corners]
                            
                            all_diff_in_neignborhood = []
                            idx = 0
                            while idx < len(neignbor_with_four_corner_scores):
                                a_s, b_s, c_s, d_s = neignbor_with_four_corner_scores[idx: idx + 4]
                                diff = round((d_s - c_s) - (b_s - a_s), 14)
                                all_diff_in_neignborhood.append(diff)
                                idx += 4
                                
                            
                            accumulated_f1_value = 0 if col == 0 else ale_values[row][col - 1]
                            accumulated_f2_value = 0 if row == 0 else ale_values[row - 1][col]
                            
                            local_ale = 0 if len(all_diff_in_neignborhood) == 0 else np.mean(all_diff_in_neignborhood)
                            
                            ale_values[row][col] = accumulated_f1_value + accumulated_f2_value + local_ale
                            
                            curr_idx += number_of_neignbor_with_four_corners                            

                    et = round(time.time() - st, 6)
                    print(f'\tcompute ale data for {output_data_sample_name}_2w_ale_{f1_feature_name}_{f2_feature_name} within {et} sec')
                    save_pdp_to_npz_2w(output_exp_dir, npz_file_path, f1_quantiles, f2_quantiles, ale_values)

         
def get_ale_and_save_score(exp_dir_path, exp_name, for_2way):
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
                try:
                    query = t['query']
                    is_for_2way = t.get('twoway') if t.get('twoway') != None else False
                    quantile_config = t.get('quantiles')
                    interval_config = t.get('intervals')
                    compute_and_save(
                        exp_dir_path, tested_sample_name, query, quantile_config, interval_config,
                        tested_sample_from_exp, tested_sample_data_source_name, for_2way and is_for_2way)
                except FileNotFoundError as e:
                    print(e)
        else:
            print(f'**no config for tested sample {tested_sample_name}')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        exp_list = sys.argv[1:]
        for_2way = '--2w' in exp_list
        if for_2way:
            exp_list = [x for x in exp_list if x != '--2w']
        for exp_name in exp_list:
            exp_dir_path = path.join(data_dir, exp_name)
            if path.isdir(exp_dir_path):
                get_ale_and_save_score(exp_dir_path, exp_name, for_2way)
            else:
                print(f'**no exp dir {exp_dir_path}')
