import time
import os
import os.path as path
import sys
import yaml
import json
import math
import numpy as np
from getting_data import load_sample
from s2search_score_pipelining import get_scores
from s2search_score_pdp import compute_pdp

model_dir = './s2search_data'
data_dir = str(path.join(os.getcwd(), 'pipelining'))
ranker = None
data_loading_line_limit = 1000

def read_conf(exp_dir_path):
    conf_path = path.join(exp_dir_path, 'conf.yml')
    with open(str(conf_path), 'r') as f:
        conf = yaml.safe_load(f)
        return conf.get('description'), conf.get('samples'), conf.get('sample_from_other_exp'),


def save_pdp_to_npz(exp_dir_path, npz_file_path, data):
    scores_dir = path.join(exp_dir_path, 'scores')
    if not path.exists(str(scores_dir)):
        os.mkdir(str(scores_dir))
    print(f'\tsave PDP data to {npz_file_path}')
    np.savez_compressed(npz_file_path, data)

def compute_diagonal_pdp(f1_name, f2_name, query, paper_data):
    print(f'\tgetting diagonal pdp for {f1_name}, {f2_name}')
    st = time.time()
    variant_data = []
    for i in range(len(paper_data)):
        # pick the i th features value
        f1_value_that_is_used = paper_data[i][f1_name]
        f2_value_that_is_used = paper_data[i][f2_name]
        # replace it to all papers
        for p in paper_data:
            new_data = {**p}
            new_data[f1_name] = f1_value_that_is_used
            new_data[f2_name] = f2_value_that_is_used
            variant_data.append(new_data)

    scores = get_scores(query, variant_data, task_name='hs')

    scores_split = np.array_split(scores, len(paper_data))
    pdp_value = [np.mean(x) for x in scores_split]

    et = round(time.time() - st, 6)
    print(f'\tcompute {len(scores)} diagonal pdp within {et} sec')
    return pdp_value

def get_pdp_data_if_exist(output_exp_dir, output_data_sample_name, 
                          data_exp_name, data_sample_name, f1_name, f2_name, query, paper_data):
    f1_pdp_data = None
    f2_pdp_data = None
    f1_f2_diagonal_pdp_data = None

    f1_pdp_from_source_path = path.join(data_dir, data_exp_name, 'scores',
                                        f"{data_sample_name}_pdp_{f1_name}.npz")
    f1_pdp_from_here_path = path.join(output_exp_dir, 'scores',
                                f"{output_data_sample_name}_pdp_{f1_name}.npz")
    
    f1_data_exist = False
    if os.path.exists(f1_pdp_from_source_path):
        f1_pdp_data = np.load(f1_pdp_from_source_path)['arr_0']  
        f1_data_exist = True
                
    if os.path.exists(f1_pdp_from_here_path):
        f1_pdp_data = np.load(f1_pdp_from_here_path)['arr_0']
        f1_data_exist = True
    
    f2_pdp_from_source_path = path.join(data_dir, data_exp_name, 'scores',
                                f"{data_sample_name}_pdp_{f2_name}.npz")            
    f2_pdp_from_here_path = path.join(output_exp_dir, 'scores',
                                f"{output_data_sample_name}_pdp_{f2_name}.npz")
    
    f2_data_exist = False
    if os.path.exists(f2_pdp_from_source_path):
        f2_pdp_data = np.load(f2_pdp_from_source_path)['arr_0']
        f2_data_exist = True
                
    if os.path.exists(f2_pdp_from_here_path):
        f2_pdp_data = np.load(f2_pdp_from_here_path)['arr_0']
        f2_data_exist = True
    
    f1_f2_data_exist = False
    f1_f2_diagonal_pdp_path = path.join(output_exp_dir, 'scores',
                                f"{output_data_sample_name}_diagonal_pdp_{f1_name}_{f2_name}.npz")
    
    if os.path.exists(f1_f2_diagonal_pdp_path):
        f1_f2_diagonal_pdp_data = np.load(f1_f2_diagonal_pdp_path)['arr_0']
        f1_f2_data_exist = True
        
    if not f1_data_exist:
        f1_pdp_data = compute_pdp(paper_data, query, f1_name)
        save_pdp_to_npz(output_exp_dir, f1_pdp_from_here_path, f1_pdp_data)
    if not f2_data_exist:
        f2_pdp_data = compute_pdp(paper_data, query, f2_name)
        save_pdp_to_npz(output_exp_dir, f2_pdp_from_here_path, f2_pdp_data)
    if not f1_f2_data_exist:
        f1_f2_diagonal_pdp_data = compute_diagonal_pdp(f1_name, f2_name, query, paper_data)
        save_pdp_to_npz(output_exp_dir, f1_f2_diagonal_pdp_path, f1_f2_diagonal_pdp_data)
        
    return f1_pdp_data, f2_pdp_data, f1_f2_diagonal_pdp_data

def compute_and_save(output_exp_dir, output_data_sample_name, query, data_exp_name, data_sample_name):
    
    df = load_sample(data_exp_name, data_sample_name)
    paper_data = json.loads(df.to_json(orient='records'))
    
    data_len = len(paper_data)
    categorical_features = [
        'title', 
        'abstract',
        'venue',
        'authors'
    ]

    for f1_idx in range(len(categorical_features)):
        for f2_idx in range(f1_idx + 1, len(categorical_features)):
            f1_name = categorical_features[f1_idx]
            f2_name = categorical_features[f2_idx]
            npz_file_path = path.join(output_exp_dir, 'scores',
                                      f"{output_data_sample_name}_hs_{f1_name}_{f2_name}.npz")

            f1_pdp_data, f2_pdp_data, f1_f2_diagonal_pdp_data = \
                get_pdp_data_if_exist(output_exp_dir, output_data_sample_name, data_exp_name,
                                      data_sample_name, f1_name, f2_name, query, paper_data)
            
            if not os.path.exists(npz_file_path):
                numerator = 0
                denominator = 0
                for i in range(data_len):
                    f1_pdp_point = f1_pdp_data[i]
                    f2_pdp_point = f2_pdp_data[i]
                    f1_and_f2_pdp_point = f1_f2_diagonal_pdp_data[i]
                    numerator += math.pow(f1_and_f2_pdp_point -
                                          f1_pdp_point - f2_pdp_point, 2)
                    denominator += math.pow(f1_and_f2_pdp_point, 2)

                h_jk = numerator / denominator
                h_jk_sqrt = math.sqrt(numerator)
                print(f'get h statistic of {f1_name} and {f2_name}      \t-> {h_jk}\t :   {h_jk_sqrt}')


def get_hstatistic_and_save_score(exp_dir_path):
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
                    compute_and_save(
                        exp_dir_path, tested_sample_name, query,
                        tested_sample_from_exp, tested_sample_data_source_name)
                except FileNotFoundError as e:
                    print(e)
        else:
            print(f'**no config for tested sample {tested_sample_name}')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        exp_list = sys.argv[1:]
        for exp_name in exp_list:
            exp_dir_path = path.join(data_dir, exp_name)
            if path.isdir(exp_dir_path):
                get_hstatistic_and_save_score(exp_dir_path)
            else:
                print(f'**no exp dir {exp_dir_path}')
