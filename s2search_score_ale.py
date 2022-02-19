from s2search.rank import S2Ranker
import time
import os
import os.path as path
import sys
import yaml
import math
import numpy as np
from getting_data import load_sample

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


def get_scores(query, paper):
    init_ranker()
    scores = ranker.score(query, paper)
    return scores


def read_conf(exp_dir_path):
    conf_path = path.join(exp_dir_path, 'conf.yml')
    with open(str(conf_path), 'r') as f:
        conf = yaml.safe_load(f)
        return conf.get('description'), conf.get('samples'), conf.get('sample_from_other_exp'),


def save_pdp_to_npz(exp_dir_path, npz_file_path, quantile, ale_result):
    scores_dir = path.join(exp_dir_path, 'scores')
    if not path.exists(str(scores_dir)):
        os.mkdir(str(scores_dir))
    print(f'\tsave ale data to {npz_file_path}')
    np.savez_compressed(npz_file_path, quantile=quantile, ale_result=ale_result)

def divide_by_percentile(df, sorted_column_name, percentile_interval):
    data_len = len(df.index)
    actual_interval = math.ceil(data_len * (percentile_interval * 0.01))
    # print(actual_interval)
    grids = []
    quantiles = []
    curr_idx = 0
    while curr_idx < data_len:
        upper_idx = curr_idx + actual_interval - 1 if curr_idx + actual_interval - 1 < data_len else data_len - 1
        grids.append({
            'lower_z': df[sorted_column_name].iloc[curr_idx],
            'upper_z': df[sorted_column_name].iloc[upper_idx],
            'neighbor': df.iloc[curr_idx: upper_idx + 1]
        })
        quantiles.append(df[sorted_column_name].iloc[upper_idx])
        curr_idx += actual_interval
    # print(len(quantiles))

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
        curr_accumulated += np.sum(all_diff_in_this) / len(all_diff_in_this)
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

    # print(paper_list[0])

    idx = 0
    while idx < len(scores):
        upper_score = scores[idx]
        lower_score = scores[idx + 1]
        diff_list.append(upper_score - lower_score)
        idx += 2
    
    return diff_list

def compute_and_save(output_exp_dir, output_data_sample_name, query, quantile_config, data_exp_name, data_sample_name):
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
            st = time.time()
            quantile_interval = quantile_config.get(feature_name) if quantile_config != None and quantile_config.get(feature_name) != None else def_quantile_config.get(feature_name) 
            grids, quantiles = divide_by_percentile(df, feature_name, quantile_interval)
            ale_result = get_ale(grids, feature_name, query)
            
            et = round(time.time() - st, 6)
            print(f'\tcompute ale data for {output_data_sample_name}_1w_ale_{feature_name} within {et} sec')
            if (feature_name == 'authors'):
                quantiles = [str(x) for x in quantiles]
            save_pdp_to_npz(output_exp_dir, npz_file_path, quantiles, ale_result)
        else:
            print(f'\t{npz_file_path} exist, should skip')
         
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
                compute_and_save(
                    exp_dir_path, tested_sample_name, query, quantile_config,
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
