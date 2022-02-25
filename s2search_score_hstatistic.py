from s2search.rank import S2Ranker
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


def pdp_for_a_point(feature_kv, query, paper_data):
    varant_paper_list = []
    for i in range(len(paper_data)):
        new_data = {**paper_data[i]}
        for key in feature_kv.keys():
            new_data[key] = feature_kv[key]
        varant_paper_list.append(new_data)

    # print()
    # for p in varant_paper_list:
    #     print(p)

    scores = get_scores(query, varant_paper_list)
    # scores = [1]

    return np.mean(scores)


def compute_and_save(output_exp_dir, output_data_sample_name, query, data_exp_name, data_sample_name):
    
    df = load_sample(data_exp_name, data_sample_name)
    paper_data = json.loads(df.to_json(orient='records'))
    
    data_len = len(paper_data)
    categorical_features = [
        'title', 
        'abstract',
        'venue',
        # 'authors'
    ]

    for f1_idx in range(len(categorical_features)):
        for f2_idx in range(f1_idx + 1, len(categorical_features)):
            f1_key = categorical_features[f1_idx]
            f2_key = categorical_features[f2_idx]
            npz_file_path = path.join(exp_dir_path, 'scores',
                                      f"{data_sample_name}_hs_{f1_key}_{f2_key}.npz")

            if not os.path.exists(npz_file_path):
                numerator = 0
                denominator = 0
                st = time.time()
                
                # time spend for 500 papers
                #   500 ^ 2 * 3 / 10000 * 7 / 60 = 8.75 min
                for i in range(data_len):
                    f1_value = paper_data[i][f1_key]
                    f2_value = paper_data[i][f2_key]

                    # print(f'the {i + 1} pdp of {f1_key} and {f2_key}')

                    f1_pdp_point = pdp_for_a_point(
                        {
                            f1_key: f1_value
                        }, query, paper_data)
                    f2_pdp_point = pdp_for_a_point(
                        {
                            f2_key: f2_value
                        }, query, paper_data)

                    f1_and_f2_pdp_point = pdp_for_a_point(
                        {
                            f1_key: f1_value,
                            f2_key: f2_value,
                        }, query, paper_data)

                    # print(f1_pdp_point, f2_pdp_point, f1_and_f2_pdp_point)

                    numerator += math.pow(f1_and_f2_pdp_point -
                                          f1_pdp_point - f2_pdp_point, 2)
                    denominator += math.pow(f1_and_f2_pdp_point, 2)

                h_jk = numerator / denominator
                h_jk_sqrt = math.sqrt(numerator)
                et = round(time.time() - st, 6)
                print(
                    f'get h statistic {h_jk}\t : {h_jk_sqrt} of {f1_key} and {f2_key} in {et} sec with {data_len} papers')


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
                query = t['query']
                compute_and_save(
                    exp_dir_path, tested_sample_name, query,
                    tested_sample_from_exp, tested_sample_data_source_name)
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
