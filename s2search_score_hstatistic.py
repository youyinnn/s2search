from s2search.rank import S2Ranker
import time
import os
import os.path as path
import sys
import yaml
import json
import math
import numpy as np

model_dir = './s2search_data'
data_dir = str(path.join(os.getcwd(), 'pipelining'))
ranker = None
data_loading_line_limit = 1000


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
        if new_data.get('id') != None:
            del new_data['id']
        if new_data.get('s2_id') != None:
            del new_data['s2_id']
        for key in feature_kv.keys():
            new_data[key] = feature_kv[key]
        varant_paper_list.append(new_data)

    # print()
    # for p in varant_paper_list:
    #     print(p)

    scores = get_scores(query, varant_paper_list)
    # scores = [1]

    return np.mean(scores)


def compute_and_save(exp_dir_path, data_sample_name, query, paper_data):
    data_len = len(paper_data)
    categorical_features = [
        'title', 'abstract',
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

    sample_file_list = [f for f in os.listdir(exp_dir_path) if path.isfile(
        path.join(exp_dir_path, f)) and f.endswith('.data')]

    for data_sample_file_name in sample_file_list:
        paper_data = []
        with open(path.join(exp_dir_path, data_sample_file_name)) as f:
            lines = f.readlines()
            for line in lines:
                paper_data.append(json.loads(line.strip()))
        data_sample_name = data_sample_file_name.replace(
            '.data', '').replace('.data', '')
        task = sample_configs.get(data_sample_name)
        if task != None:
            print(f'computing pdp for {data_sample_file_name}')
            for t in task:
                query = t['query']
                compute_and_save(
                    exp_dir_path, data_sample_name, query, paper_data)
        else:
            print(f'**no config for data file {data_sample_file_name}')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        exp_list = sys.argv[1:]
        for exp_name in exp_list:
            exp_dir_path = path.join(data_dir, exp_name)
            if path.isdir(exp_dir_path):
                get_hstatistic_and_save_score(exp_dir_path)
            else:
                print(f'**no exp dir {exp_dir_path}')
