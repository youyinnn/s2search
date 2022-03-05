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


def save_pdp_to_npz(exp_dir_path, npz_file_path, *args):
    scores_dir = path.join(exp_dir_path, 'scores')
    if not path.exists(str(scores_dir)):
        os.mkdir(str(scores_dir))
    print(f'\tsave PDP data to {npz_file_path}')
    np.savez_compressed(npz_file_path, *args)

def save_pdp_to_npz_for_numerical(exp_dir_path, npz_file_path, pdp_value, feature_value):
    scores_dir = path.join(exp_dir_path, 'scores')
    if not path.exists(str(scores_dir)):
        os.mkdir(str(scores_dir))
    print(f'\tsave PDP data to {npz_file_path}')
    np.savez_compressed(npz_file_path, pdp_value=pdp_value, feature_value=feature_value)

def get_feature_space_and_count(df, fk):
    count_map = {}
    values = list(df[fk])
    if fk == 'authors':
        values = [json.dumps(v) for v in values]
    for v in values:
        if count_map.get(v) == None:
            count_map[v] = 1
        else:
            count_map[v] += 1

    return list(dict.fromkeys(values)), count_map

def compute_2w_and_save(output_exp_dir, output_data_sample_name, query, data_exp_name, data_sample_name):
    feature_names = [
        'title', 
        'abstract', 'venue', 
        'authors', 
        'year', 
         'n_citations'
    ]
    
    for f1_idx in range(len(feature_names)):
        for f2_idx in range(f1_idx + 1, len(feature_names)):
            f1_key = feature_names[f1_idx]
            f2_key = feature_names[f2_idx]
            
            npz_file_path = path.join(output_exp_dir, 'scores',
                                      f"{output_data_sample_name}_pdp2w_{f1_key}_{f2_key}.npz")

            if not os.path.exists(npz_file_path):
                print(f'computing 2w pdp for {output_data_sample_name} {f1_key} {f2_key}')

                f1_df = load_sample(data_exp_name, data_sample_name, sort=f1_key, del_f = ['s2_id'], rank_f=get_scores, query=query)
                f2_df = load_sample(data_exp_name, data_sample_name, sort=f2_key, del_f = ['s2_id'], rank_f=get_scores, query=query)
                
                paper_data = json.loads(f1_df.to_json(orient='records'))
                paper_len = len(paper_data)
                
                f1_feature_space, f1_feature_count_map = get_feature_space_and_count(f1_df, f1_key);
                f2_feature_space, f2_feature_count_map = get_feature_space_and_count(f2_df, f2_key);

                variant_data = []
                
                # print(f1_feature_count_map)
                # print(f2_feature_count_map)

                print(f'compute {paper_len * paper_len * paper_len} but actually computing {len(f1_feature_space) * len(f2_feature_space) * len(paper_data)}')
                
                # compressing
                for f1_value in f1_feature_space:
                    f1_value = f1_value if f1_key != 'authors' else json.loads(f1_value)
                    for f2_value in f2_feature_space:
                        f2_value = f2_value if f2_key != 'authors' else json.loads(f2_value)
                        for p in paper_data:
                            varient = {**p}
                            varient[f1_key] = f1_value
                            varient[f2_key] = f2_value
                            variant_data.append(varient)
                            
                # print(len(variant_data))
                scores = get_scores(query, variant_data, task_name='pdp-2w')
                vd_split = np.array_split(variant_data, len(f1_feature_space) * len(f2_feature_space))
                
                # for vd in vd_split:
                #     print()
                #     for p in vd:
                #         print(p['id'], p[f1_key], p[f2_key])
                
                scores_split = np.array_split(scores, len(f1_feature_space) * len(f2_feature_space))
                
                decompressing_data = []
                decompressing_data_scores = []
                
                # decompressing
                split_idx = 0
                for f1_value in f1_feature_space:
                    f1_value_count = f1_feature_count_map[f1_value]
                    for i in range(f1_value_count):
                        for f2_value in f2_feature_space:
                            f2_value_count = f2_feature_count_map[f2_value]
                            varient_for_f1_f2 = vd_split[split_idx]
                            scores_for_varient_f1_f2 = scores_split[split_idx]
                            for j in range(f2_value_count):
                                decompressing_data.extend(varient_for_f1_f2)
                                decompressing_data_scores.extend(scores_for_varient_f1_f2)
                            # move to next col
                            split_idx += 1
                        # go back to this rowu
                        split_idx -= len(f2_feature_space)

                    # move to next row
                    split_idx += len(f2_feature_space)
                        
                # idx = 0
                # for i in range(len(decompressing_data)):
                #     p = decompressing_data[i]
                #     if idx % (paper_len * paper_len) == 0:
                #         print()
                #     print(f"{p['id']}\t", p[f1_key], p[f2_key], decompressing_data_scores[i])
                #     idx += 1
                
                pdp_2w_value = np.zeros([paper_len, paper_len])
                
                curr_idx = 0
                for col in range(paper_len):
                    for row in range(paper_len):
                        # print(row, col, decompressing_data_scores[curr_idx: curr_idx + paper_len])
                        pdp_2w_value[row][col] = np.mean(decompressing_data_scores[curr_idx: curr_idx + paper_len])
                        curr_idx += paper_len
                
                save_pdp_to_npz(exp_dir_path, npz_file_path, pdp_2w_value)
            else:
                print(f'\t{npz_file_path} exist, should skip')

def save_original_scores(output_exp_dir, output_data_sample_name, query, paper_data):
    npz_file_path = path.join(output_exp_dir, 'scores', f"{output_data_sample_name}_pdp_original.npz")
    print(f'save original scores {npz_file_path}')
    if not os.path.exists(npz_file_path):
        scores = get_scores(query, paper_data, task_name='pdp-1w')
        save_pdp_to_npz(output_exp_dir, npz_file_path, scores)
    else:
        print(f'\t{npz_file_path} exist, should skip')
    
def compute_pdp(paper_data, query, feature_name):
    data_len = len(paper_data)
    # pdp_value = []
    print(f'\tgetting pdp for {feature_name}')
    st = time.time()
    if feature_name == 'year' or feature_name == 'n_citations':
        value_map = {}
        for p in paper_data:
            value = p[feature_name]
            vinm = value_map.get(value)
            if vinm == None:
                value_map[value] = 1
            else:
                value_map[value] += 1
        
        # print(value_map)

        sorted_values = list(value_map.keys())
        sorted_values.sort()

        variant_data = []
        for value_that_is_used in sorted_values:
            for p in paper_data:
                new_data = {**p}
                new_data[feature_name] = value_that_is_used
                variant_data.append(new_data)

        print(f'get {len(paper_data) * len(paper_data)} numerical pdp but actually computing {len(sorted_values) * len(paper_data)}')
        scores = get_scores(query, variant_data, task_name='pdp-numerical')
        
        scores_split = np.array_split(scores, len(sorted_values))
        # pdp_value = [np.mean(x) for x in scores_split]
        
        decompressed_scores_split = []
        
        idx = 0
        for value in sorted_values:
            count = value_map[value]
            # pdp_v = np.mean(scores_split[idx])
            for i in range(count):
                # pdp_value.append(pdp_v)
                decompressed_scores_split.append(scores_split[idx])
            idx += 1
        scores_split = decompressed_scores_split
    else:
        variant_data = []
        for i in range(data_len):
            # pick the i th features value
            value_that_is_used = paper_data[i][feature_name]
            # replace it to all papers
            for p in paper_data:
                new_data = {**p}
                new_data[feature_name] = value_that_is_used
                variant_data.append(new_data)

        scores = get_scores(query, variant_data, task_name='pdp-categorical')
        scores_split = np.array_split(scores, len(paper_data))
        # pdp_value = [np.mean(x) for x in scores_split]
        
    et = round(time.time() - st, 6)
    print(f'\tcompute {len(scores)} pdp within {et} sec')
    return scores_split

def compute_and_save(output_exp_dir, output_data_sample_name, query, data_exp_name, data_sample_name):
    df = load_sample(data_exp_name, data_sample_name)
    paper_data = json.loads(df.to_json(orient='records'))
    # save_original_scores(output_exp_dir, output_data_sample_name, query, paper_data)
    categorical_features = ['title', 'abstract', 'venue', 'authors', 'year', 'n_citations']
    for feature_name in categorical_features:
        npz_file_path = path.join(output_exp_dir, 'scores',
                                  f"{output_data_sample_name}_pdp_{feature_name}.npz")
        if not os.path.exists(npz_file_path):
            pdp_value = compute_pdp(paper_data, query, feature_name)
            if feature_name == 'year' or feature_name == 'n_citations':
                feature_values = [paper[feature_name] for paper in paper_data] 
                save_pdp_to_npz(output_exp_dir, npz_file_path, pdp_value, feature_values)
            else:
                save_pdp_to_npz(output_exp_dir, npz_file_path, pdp_value)
        else:
            print(f'\t{npz_file_path} exist, should skip')


def get_pdp_and_save_score(exp_dir_path, exp_name, for_2way):
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
            print(f'computing pdp for {tested_sample_name}')
            for t in task:
                try:
                    query = t['query']
                    is_for_2way = t.get('twoway') if t.get('twoway') != None else False
                    
                    if for_2way and is_for_2way:
                        compute_2w_and_save(
                            exp_dir_path, tested_sample_name, query,
                            tested_sample_from_exp, tested_sample_data_source_name
                        )
                    elif not for_2way:
                        compute_and_save(
                            exp_dir_path, tested_sample_name, query,
                            tested_sample_from_exp, tested_sample_data_source_name
                        )
                    else:
                        print(f'{tested_sample_name} is not for 2 way pdp')
                except FileNotFoundError as e:
                    print(e)
        else:
            print(f'**no config for tested sample {tested_sample_name}')

def is_numerical_feature(feature_name):        
    return True if feature_name == 'year' or feature_name == 'n_citations' else False

def pdp_based_importance(pdp_value, feature_name):
    pdp_np = np.array(pdp_value, dtype='float64')
    # if is_numerical_feature(feature_name):
    if True:
        _k = len(pdp_value)
        _mean = np.mean(pdp_value)
        pdp_np -= _mean
        pdp_np *= pdp_np
        return round(np.sqrt(np.sum(pdp_np) / (_k - 1)), 10)      
    else:
        return (np.max(pdp_value) - np.min(pdp_value)) / 4

if __name__ == '__main__':
    if len(sys.argv) > 1:
        exp_list = sys.argv[1:]
        for_2way = '--2w' in exp_list    
        if for_2way:
            exp_list = [x for x in exp_list if x != '--2w']

        for exp_name in exp_list:
            exp_dir_path = path.join(data_dir, exp_name)
            if path.isdir(exp_dir_path):
                get_pdp_and_save_score(exp_dir_path, exp_name, for_2way)
            else:
                print(f'**no exp dir {exp_dir_path}')
