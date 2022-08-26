import json
import os
import numpy as np
import yaml
import pandas as pd
from ranker_helper import get_scores

feature_key_list = ['title', 'abstract',
                    'venue', 'authors', 'year', 'n_citations']
categorical_feature_key_list = ['title', 'abstract', 'venue', 'authors']

masking_option_keys = [
    "t",
    "abs",
    "v",
    "au",
    "y",
    "c",
    "tabs",
    "tabsv",
    "tabsvau",
    "tabsvauy",
    "tabsvauyc",
    "tabsvauc",
    "tabsvy",
    "tabsvyc",
    "tabsvc",
    "tabsau",
    "tabsauy",
    "tabsauyc",
    "tabsauc",
    "tabsy",
    "tabsyc",
    "tabsc",
    "tv",
    "tvau",
    "tvauy",
    "tvauyc",
    "tvauc",
    "tvy",
    "tvyc",
    "tvc",
    "tau",
    "tauy",
    "tauyc",
    "tauc",
    "ty",
    "tyc",
    "tc",
    "absv",
    "absvau",
    "absvauy",
    "absvauyc",
    "absvauc",
    "absvy",
    "absvyc",
    "absvc",
    "absau",
    "absauy",
    "absauyc",
    "absauc",
    "absy",
    "absyc",
    "absc",
    "vau",
    "vauy",
    "vauyc",
    "vauc",
    "vy",
    "vyc",
    "vc",
    "auy",
    "auyc",
    "auc",
    "yc",
]


def get(exp_name, sample_name):

    work_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir_path = os.path.join(work_dir, 'pipelining', exp_name)

    conf_path = os.path.join(exp_dir_path, 'conf.yml')
    with open(conf_path, 'r') as f:
        conf = yaml.safe_load(f)
        samples_config = conf.get('samples')

    # sample_file_list = samples_config.keys()
    print(f'Got sample data: {sample_name} {exp_dir_path}')

    # preparing data
    score_dir = os.path.join(exp_dir_path, 'scores')

    sample_data_and_config = []
    sample_task_list = samples_config[sample_name]

    if (type(sample_task_list) == dict):
        # new config
        sample_task_list = sample_task_list['masking']
        for t in sample_task_list:
            t['masking_option_keys'] = masking_option_keys.copy()

    t_count = 0
    for task in sample_task_list:
        t_count += 1
        sample_query = task['query']
        sample_masking_option_keys = task['masking_option_keys']

        sample_origin_npy = np.load(os.path.join(
            score_dir, f'{exp_name}_{sample_name}_t{t_count}_origin.npz'))['arr_0']

        sample_feature_masking_npy = []
        for key in sample_masking_option_keys:
            sample_feature_masking_npy.append(np.load(os.path.join(
                score_dir, f'{exp_name}_{sample_name}_t{t_count}_{key}.npz'))['arr_0'])
        if len(sample_feature_masking_npy) > 0:
            feature_stack = np.stack((sample_feature_masking_npy))
        else:
            feature_stack = np.stack(([sample_origin_npy]))
            sample_masking_option_keys.append('_origin')

        sample_data_and_config.append({
            'sample_and_task_name': f'{sample_name}',
            'task_number': t_count,
            'query': sample_query,
            'origin': sample_origin_npy,
            'feature_stack': feature_stack,
            'masking_option_keys': sample_masking_option_keys
        })

    return sample_data_and_config


def load_sample(exp_name, sample_name, sort=None, del_f=['id', 's2_id'],
                rank_f=None, query=None, author_as_str=False, task_name=None, not_df=False):

    data = []
    pipelining_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'pipelining')

    with open(os.path.join(pipelining_dir, exp_name, f'{sample_name}.data')) as f:
        lines = f.readlines()
        for line in lines:
            jso = json.loads(line.strip())
            if del_f != None:
                for k in del_f:
                    if jso.get(k) != None:
                        del jso[k]
            if author_as_str:
                jso['authors'] = json.dumps(jso['authors'])
            data.append(jso)

    if not_df:
        return data

    if sort != None:
        if sort == 'year' or sort == 'n_citations':
            data.sort(key=lambda x: x[sort])
        else:
            # masking
            df = pd.read_json(
                f"[{','.join(list(map(lambda x: json.dumps(x), data)))}]")
            dfd = df.drop([x for x in feature_key_list if x != sort], axis=1)
            masked_paper = json.loads(dfd.to_json(orient='records'))
            # ranking
            masked_scores = rank_f(query, masked_paper, task_name=task_name)
            scores_df = pd.DataFrame(data={'score': masked_scores})
            return pd.concat([df, scores_df], axis=1).sort_values(by=['score']).astype({
                'title': 'category',
                'abstract': 'category',
                'venue': 'category',
                'authors': 'category' if author_as_str else 'object',
            })

    return pd.read_json(f"[{','.join(list(map(lambda x: json.dumps(x), data)))}]").astype({
        'title': 'category',
        'abstract': 'category',
        'venue': 'category',
        'authors': 'category' if author_as_str else 'object',
    })


def read_conf(exp_dir_path):
    conf_path = os.path.join(exp_dir_path, 'conf.yml')
    with open(str(conf_path), 'r') as f:
        conf = yaml.safe_load(f)
        return conf.get('description'), conf.get('samples'), conf.get('sample_from_other_exp'),


def remove_duplicate(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def get_categorical_encoded_data(data_exp_name, data_sample_name, query, paper_data=None):
    if paper_data == None:
        paper_data = load_sample(data_exp_name, data_sample_name, not_df=True)

    categorical_name = {}

    for i in range(len(feature_key_list)):
        feature_name = feature_key_list[i]
        if feature_name in categorical_feature_key_list:
            df = load_sample(data_exp_name, data_sample_name,
                             query=query, sort=feature_name, rank_f=get_scores)
            if feature_name == 'authors':
                l = [json.dumps(x) for x in df[feature_name]]
            else:
                l = list(df[feature_name])
            categorical_name[i] = remove_duplicate(l)

    categorical_name_map = {}
    for i in range(len(feature_key_list)):
        feature_name = feature_key_list[i]
        if feature_name in categorical_feature_key_list:
            categorical_name_map[i] = {}
            values = categorical_name[i]
            for j in range(len(values)):
                value = values[j]
                categorical_name_map[i][value] = j

    # encoding data
    for i in range(len(paper_data)):
        paper_data[i] = [
            categorical_name_map[0][paper_data[i]['title']
                                    ], categorical_name_map[1][paper_data[i]['abstract']],
            categorical_name_map[2][paper_data[i]['venue']], categorical_name_map[3][json.dumps(
                paper_data[i]['authors'])],
            paper_data[i]['year'],
            paper_data[i]['n_citations']
        ]

    paper_data = np.array(paper_data)

    return (categorical_name, paper_data)


def decode_paper(categorical_name, encoded_p):
    # return dict(
    #     title=categorical_name[0][int(
    #         encoded_p[0])] if encoded_p[0] != -1 else ' ',
    #     abstract=categorical_name[1][int(
    #         encoded_p[1])] if encoded_p[1] != -1 else ' ',
    #     venue=categorical_name[2][int(
    #         encoded_p[2])] if encoded_p[2] != -1 else ' ',
    #     authors=json.loads(categorical_name[3][int(
    #         encoded_p[3])]) if encoded_p[3] != -1 else [],
    #     year=encoded_p[4] if encoded_p[4] != -1 else ' ',
    #     n_citations=encoded_p[5] if encoded_p[5] != -1 else 0,
    # )
    rs = {}
    if encoded_p[0] != -1:
        rs['title'] = categorical_name[0][int(encoded_p[0])]
    if encoded_p[1] != -1:
        rs['abstract'] = categorical_name[1][int(encoded_p[1])]
    if encoded_p[2] != -1:
        rs['venue'] = categorical_name[2][int(encoded_p[2])]
    if encoded_p[3] != -1:
        rs['authors'] = json.loads(categorical_name[3][int(encoded_p[3])])
    if encoded_p[4] != -1:
        rs['year'] = encoded_p[4]
    if encoded_p[5] != -1:
        rs['n_citations'] = encoded_p[5]

    return rs
