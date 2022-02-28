import time
import os
import os.path as path
import sys
import yaml
import json
import numpy as np
from getting_data import load_sample
try:
  from s2search_score_pipelining import get_scores
except ModuleNotFoundError as e:
  print(f's2search_score_pdp', e)

model_dir = './s2search_data'
data_dir = str(path.join(os.getcwd(), 'pipelining'))
ranker = None
year_pdp_value_space = range(1960, 2023)
citation_pdp_value_space = range(0, 15000, 100)

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


def compute_2d_and_save(exp_dir_path, data_sample_name, query, paper_data):
    data_len = len(paper_data)
    categorical_features = ['title', 'abstract', 'venue', 'authors']
    for f1_idx in range(len(categorical_features)):
        for f2_idx in range(f1_idx + 1, len(categorical_features)):
            f1_key = categorical_features[f1_idx]
            f2_key = categorical_features[f2_idx]
            npz_file_path = path.join(exp_dir_path, 'scores',
                                      f"{data_sample_name}_pdp2w_{f1_key}_{f2_key}.npz")
            # print(f1_key, f2_key)

            if not os.path.exists(npz_file_path):
                paper_score_map = []
                for i in range(data_len):
                    for p in paper_data:
                        variant_data = []
                        for j in range(data_len):
                            value_that_feature_2_is_used = paper_data[i][f2_key]
                            new_data = {**p}
                            new_data[f2_key] = value_that_feature_2_is_used
                            value_that_feature_1_is_used = paper_data[j][f1_key]
                            new_data[f1_key] = value_that_feature_1_is_used
                            variant_data.append(new_data)

                    # for p in variant_data:
                    #     print(p)
                    # print()

                    st = time.time()
                    scores = get_scores(query, variant_data)
                    et = round(time.time() - st, 6)

                    print(f'done with {i + 1},{j + 1} within {et} sec')
                    paper_score_map.append(scores.tolist())

                # print(paper_score_map)
                save_pdp_to_npz(exp_dir_path, npz_file_path, paper_score_map)
            else:
                print(f'\t{npz_file_path} exist, should skip')

    # TODO: numerical features 2-way pdp

def save_original_scores(output_exp_dir, output_data_sample_name, query, paper_data):
    npz_file_path = path.join(output_exp_dir, 'scores', f"{output_data_sample_name}_pdp_original.npz")
    print(f'save original scores {npz_file_path}')
    if not os.path.exists(npz_file_path):
        scores = get_scores(query, paper_data)
        save_pdp_to_npz(output_exp_dir, npz_file_path, scores)
    else:
        print(f'\t{npz_file_path} exist, should skip')
    
def compute_pdp(paper_data, query, feature_name):
    data_len = len(paper_data)
    pdp_value = []
    print(f'\tgetting pdp for {feature_name}')
    st = time.time()
    if feature_name == 'year' or feature_name == 'n_citations':
        if feature_name == 'year':
            rg = year_pdp_value_space
        else:
            rg = citation_pdp_value_space
        for i in rg:
            value_that_is_used = i
            variant_data = []
            for p in paper_data:
                new_data = {**p}
                new_data[feature_name] = value_that_is_used
                variant_data.append(new_data)

            scores = get_scores(query, variant_data)
            pdp_value.append(np.mean(scores))
    else:
        for i in range(data_len):
            # pick the i th features value
            value_that_is_used = paper_data[i][feature_name]
            variant_data = []
            # replace it to all papers
            for p in paper_data:
                new_data = {**p}
                new_data[feature_name] = value_that_is_used
                variant_data.append(new_data)

            # get the scores
            scores = get_scores(query, variant_data)
            pdp_value.append(np.mean(scores))
        
    et = round(time.time() - st, 6)
    print(f'\tcompute {len(scores)} pdp within {et} sec')
    return pdp_value

def compute_and_save(output_exp_dir, output_data_sample_name, query, data_exp_name, data_sample_name):
    df = load_sample(data_exp_name, data_sample_name)
    paper_data = json.loads(df.to_json(orient='records'))
    save_original_scores(output_exp_dir, output_data_sample_name, query, paper_data)
    categorical_features = ['title', 'abstract', 'venue', 'authors', 'year', 'n_citations']
    for feature_name in categorical_features:
        npz_file_path = path.join(output_exp_dir, 'scores',
                                  f"{output_data_sample_name}_pdp_{feature_name}.npz")
        if not os.path.exists(npz_file_path):
            pdp_value = compute_pdp(paper_data, query, feature_name)
            save_pdp_to_npz(output_exp_dir, npz_file_path, pdp_value)
        else:
            print(f'\t{npz_file_path} exist, should skip')


def get_pdp_and_save_score(exp_dir_path, exp_name, is2d):
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

def is_numerical_feature(feature_name):        
    return True if feature_name == 'year' or feature_name == '' else False

def pdp_based_importance(pdp_value, feature_name):
    pdp_np = np.array(pdp_value, dtype='float64')
    if is_numerical_feature(feature_name):
        _k = len(pdp_value)
        _mean = np.mean(pdp_value)
        pdp_np -= _mean
        pdp_np *= pdp_np
        return np.sqrt(np.sum(pdp_np) / (_k - 1))         
    else:
        return (np.max(pdp_value) - np.min(pdp_value)) / 4

if __name__ == '__main__':
    if len(sys.argv) > 1:
        is2d = False
        if len(sys.argv) > 2 and sys.argv[1] == '--2w':
            is2d = True
            exp_list = sys.argv[2:]
        else:
            exp_list = sys.argv[1:]
        for exp_name in exp_list:
            exp_dir_path = path.join(data_dir, exp_name)
            if path.isdir(exp_dir_path):
                get_pdp_and_save_score(exp_dir_path, exp_name, is2d)
            else:
                print(f'**no exp dir {exp_dir_path}')
