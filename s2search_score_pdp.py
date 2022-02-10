from ntpath import join
from s2search.rank import S2Ranker
import time
import os
import os.path as path
import sys
import yaml
import json
import shutil
import zlib
from multiprocessing import Pool
import numpy as np
import feature_masking as fm

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


def save_pdp_to_npz(exp_dir_path, data_sample_name, feature_name, data):
    scores_dir = path.join(exp_dir_path, 'scores')
    if not path.exists(str(scores_dir)):
        os.mkdir(str(scores_dir))
    npz_file_path = path.join(exp_dir_path, 'scores',
                              f"{data_sample_name}_pdp_{feature_name}")
    print(f'\tsave PDP data to {npz_file_path}')
    np.savez_compressed(npz_file_path, data)


def compute_and_save(exp_dir_path, data_sample_name, query, paper_data):
    data_len = len(paper_data)
    categorical_features = ['title', 'abstract', 'venue', 'authors']
    for feature_name in categorical_features:
        npz_file_path = path.join(exp_dir_path, 'scores',
                                  f"{data_sample_name}_pdp_{feature_name}.npz")
        if not os.path.exists(npz_file_path):
            pdp_value = []
            print(f'\tgetting score for {feature_name}')
            st = time.time()
            for i in range(data_len):
                value_that_is_used = paper_data[i][feature_name]
                variant_data = []
                for p in paper_data:
                    new_data = {**p}
                    del new_data['id']
                    del new_data['s2_id']
                    new_data[feature_name] = value_that_is_used
                    variant_data.append(new_data)

                scores = get_scores(query, variant_data)
                pdp_value.append(np.mean(scores))

            et = round(time.time() - st, 6)
            save_pdp_to_npz(exp_dir_path, data_sample_name,
                            feature_name, pdp_value)
            print(
                f'\tcompute {len(scores)} scores within {et} sec')
        else:
            print(f'\t{npz_file_path} exist, should skip')

    numerical_features_and_range = [
        ['year', range(1800, 2050)],
        ['n_citations', range(0, 11000, 50)]
    ]

    for feature_and_range in numerical_features_and_range:
        pdp_value = []
        feature_name, rg = feature_and_range
        npz_file_path = path.join(exp_dir_path, 'scores',
                                  f"{data_sample_name}_pdp_{feature_name}.npz")
        if not os.path.exists(npz_file_path):
            print(f'\tgetting score for {feature_name}')
            st = time.time()
            for i in rg:
                value_that_is_used = i
                variant_data = []
                for p in paper_data:
                    new_data = {**p}
                    del new_data['id']
                    del new_data['s2_id']
                    new_data[feature_name] = value_that_is_used
                    variant_data.append(new_data)

                scores = get_scores(query, variant_data)
                pdp_value.append(np.mean(scores))

            et = round(time.time() - st, 6)
            save_pdp_to_npz(exp_dir_path, data_sample_name,
                            feature_name, pdp_value)
            print(
                f'\tcompute {len(scores)} scores within {et} sec')
        else:
            print(f'\t{npz_file_path} exist, should skip')


def get_pdp_and_save_score(exp_dir_path):
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
    pass


if __name__ == '__main__':
    if len(sys.argv) > 1:
        exp_list = sys.argv[1:]
        for exp_name in exp_list:
            exp_dir_path = path.join(data_dir, exp_name)
            if path.isdir(exp_dir_path):
                get_pdp_and_save_score(exp_dir_path)
