import time
import os
import sys
import yaml, json
import numpy as np
from s2search_score_pipelining import read_conf
from getting_data import load_sample, feature_key_list, categorical_feature_key_list
from ranker_helper import get_scores
from s2search_score_pdp import save_pdp_to_npz
from anchor import anchor_tabular
import pytz
import datetime
utc_tz = pytz.timezone('America/Montreal')

data_dir = str(os.path.join(os.getcwd(), 'pipelining'))

def get_class(score):
    if score <= -17:
        return '(,-17]'
    elif score <= -10:
        return '(-17, -10]'
    elif score <= -5:
        return '(-10, -5]'    
    elif score <= 0:
        return '(-5, <0]'  
    elif score <= 3:
        return '(0, 3]'
    elif score <= 5:
        return '(3, 5]'
    elif score <= 6:
        return '(5, 6]'
    elif score <= 7:
        return '(6, 7]'
    elif score <= 8:
        return '(7, 8]'
    elif score <= 9:
        return '(8, 9]'
    else:
        return '(9,)'
    
def remove_duplicate(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def decode_paper(categorical_name, encoded_p):
    return dict(
        title=categorical_name[0][encoded_p[0]],
        abstract=categorical_name[1][encoded_p[1]],
        venue=categorical_name[2][encoded_p[2]],
        authors=json.loads(categorical_name[3][encoded_p[3]]),
        year=encoded_p[4],
        n_citations=encoded_p[5],
    )
    

def compute_and_save(output_exp_dir, output_data_sample_name, query, rg, data_exp_name, data_sample_name):

    metrics_npz_file = os.path.join(output_exp_dir, 'scores', f'{output_data_sample_name}_metrics_{rg[0]}_{rg[1]}.npz')

    st = time.time()
    ts = datetime.datetime.now(tz=utc_tz).strftime("%m/%d/%Y, %H:%M:%S")
    print(f'[{ts}] start anchro metrics')
    
    categorical_name = {}
    categorical_name_file = os.path.join(output_exp_dir, 'scores', f'{output_data_sample_name}_categorical_name.npz')

    if not os.path.exists(categorical_name_file):
        # get categorical_name
        for i in range(len(feature_key_list)):
            feature_name = feature_key_list[i]
            if feature_name in categorical_feature_key_list:
                df = load_sample(data_exp_name, data_sample_name, query=query, sort=feature_name, rank_f=get_scores)
                if feature_name == 'authors':
                    l = [json.dumps(x) for x in df[feature_name]]
                else:
                    l = list(df[feature_name])
                categorical_name[i] = remove_duplicate(l)
                
        save_pdp_to_npz('.', categorical_name_file, title=categorical_name[0], abstract=categorical_name[1], \
        venue=categorical_name[2], authors=categorical_name[3])
    else:
        load = np.load(categorical_name_file)
        categorical_name[0] = load['title']
        categorical_name[1] = load['abstract']
        categorical_name[2] = load['venue']
        categorical_name[3] = load['authors']    
        
    df = load_sample(data_exp_name, data_sample_name)
    paper_data = json.loads(df.to_json(orient='records'))

    y_pred_file = os.path.join('.', 'scores', f'{data_sample_name}_y_pred.npz')
    if not os.path.exists(y_pred_file):
        y_pred = get_scores(query, paper_data)
        save_pdp_to_npz('.', y_pred_file, y_pred)
    else:
        y_pred = np.load(y_pred_file)['arr_0']
    
    # make class_name
    class_name = ['(,-17]','(-17, -10]','(-10, -5]','(-5, <0]','(0, 3]','(3, 5]','(5, 6]','(6, 7]','(7, 8]','(8, 9]','(9,)']

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
            categorical_name_map[0][paper_data[i]['title']], categorical_name_map[1][paper_data[i]['abstract']],
            categorical_name_map[2][paper_data[i]['venue']], categorical_name_map[3][json.dumps(paper_data[i]['authors'])],
            paper_data[i]['year'],
            paper_data[i]['n_citations']
        ]
        
    paper_data = np.array(paper_data)

    explainer = anchor_tabular.AnchorTabularExplainer(
        class_name,
        feature_key_list,
        paper_data,
        categorical_name)
    
    def pred_fn(x):
        predictions = get_scores(query, [decode_paper(categorical_name, p) for p in x], ptf = False)
        encoded_pred = [class_name.index(get_class(pp)) for pp in predictions]
        return np.array(encoded_pred)
    
    metrics = dict(
        title=0,
        abstract=0,
        venue=0,
        authors=0,
        year=0,
        n_citations=0,
    )

    for i in range(len(paper_data[rg[0]: rg[1]])):
        exp = explainer.explain_instance(paper_data[i], pred_fn, threshold=0.95, batch_size=80)
        for name in exp.names():
            for feature_name in metrics.keys():
                if name.startswith(f'{feature_name} = '):
                    metrics[feature_name] += 1


    save_pdp_to_npz('.', metrics_npz_file, [metrics[f] for f in feature_key_list])
    
    ts = datetime.datetime.now(tz=utc_tz).strftime("%m/%d/%Y, %H:%M:%S")
    
    print(metrics)
    
    print(f'[{ts}] end anchro metrics witin {round(time.time() - st, 6)}')

def get_anchor_metrics(exp_dir_path):
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
                    range = t['range']
                    compute_and_save(
                        exp_dir_path, tested_sample_name, query, range,
                        tested_sample_from_exp, tested_sample_data_source_name)
                except FileNotFoundError as e:
                    print(e)
        else:
            print(f'**no config for tested sample {tested_sample_name}')
            
            
if __name__ == '__main__':
    if len(sys.argv) > 1:
        exp_list = sys.argv[1:]
        for exp_name in exp_list:
            exp_dir_path = os.path.join(data_dir, exp_name)
            if os.path.isdir(exp_dir_path):
                get_anchor_metrics(exp_dir_path)
            else:
                print(f'**no exp dir {exp_dir_path}')