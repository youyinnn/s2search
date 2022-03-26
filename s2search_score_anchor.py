import time
import os
import sys
import yaml, json
import numpy as np
from getting_data import read_conf
from getting_data import load_sample, feature_key_list, get_categorical_encoded_data, decode_paper
from ranker_helper import get_scores
from s2search_score_pdp import save_pdp_to_npz
from anchor import anchor_tabular
import pytz
import datetime
import logging
import psutil

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
    
def get_time_str():
    return datetime.datetime.now(tz=utc_tz).strftime("%m/%d/%Y, %H:%M:%S")

def metrics_to_str(metrics):
    return ', '.join([f'{feature_name}: {len(metrics[feature_name])}' for feature_name in metrics.keys()])

def compute_and_save(output_exp_dir, output_data_sample_name, query, rg, data_exp_name, data_sample_name):    
    metrics_npz_file = os.path.join(output_exp_dir, 'scores', f'{output_data_sample_name}_metrics_{rg[0]}_{rg[1]}.npz')

    st = time.time()
    logging.info(f'\n[{get_time_str()}] start anchor metrics')
    
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
    
    categorical_name, paper_data = get_categorical_encoded_data(data_exp_name, data_sample_name, query, paper_data)
    
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
        title=[],
        abstract=[],
        venue=[],
        authors=[],
        year=[],
        n_citations=[],
    )
    
    previous_work_idx = rg[0]
    if os.path.exists(metrics_npz_file):
        previous_data = np.load(metrics_npz_file)
        metrics = dict(
            title=list(previous_data['title']),
            abstract=list(previous_data['abstract']),
            venue=list(previous_data['venue']),
            authors=list(previous_data['authors']),
            year=list(previous_data['year']),
            n_citations=list(previous_data['n_citations']),
        )
        previous_work_idx = previous_data['idx'][0] + 1

    sst = time.time()
    logging.info(f'[{get_time_str()}] start computing anchor from index: {previous_work_idx} to {rg[1] - 1}')
    count = 0
    for i in range(previous_work_idx, rg[1]):
        exp = explainer.explain_instance(paper_data[i], pred_fn, threshold=0.99, tau=0.8)
        
        previous_single_precision = 0
        for j in range(len(exp.names())):
            name = exp.names()[j]
            current_single_precision = exp.precision(j) - previous_single_precision
            previous_single_precision = exp.precision(j)
            for feature_name in metrics.keys():
                if name.startswith(f'{feature_name} = '):
                    metrics[feature_name].append(current_single_precision)
        
        count += 1
        
        if count % 10 == 0:
            ett = round(time.time() - sst, 6)
            logging.info(f'[{get_time_str()}] ({i} / {rg[1] - 1}) {metrics_to_str(metrics)} within ({ett} total / {round(ett / count, 4)} average)')
            save_pdp_to_npz('.', metrics_npz_file, 
                title=metrics['title'],
                abstract=metrics['abstract'],
                venue=metrics['venue'],
                authors=metrics['authors'],
                year=metrics['year'],
                n_citations=metrics['n_citations'],
                idx=[i]
            )
    
    logging.info(f'[{get_time_str()}] end anchor metrics witin {round(time.time() - st, 6)}')

def get_anchor_metrics(exp_dir_path, sample_name, task_number, cpu_number):

    if sys.platform != "darwin":
        p = psutil.Process()
        worker = int(cpu_number)
        print(f"Child #{worker}: {p}, affinity {p.cpu_affinity()}", flush=True)
        p.cpu_affinity([worker])
        print(f"Child #{worker}: Set my affinity to {worker}, affinity now {p.cpu_affinity()}", flush=True)

    des, sample_configs, sample_from_other_exp = read_conf(exp_dir_path)
    
    if sample_name in sample_from_other_exp.keys():
        other_exp_name, data_file_name = sample_from_other_exp.get(sample_name)
        tested_sample_config = {'exp_name': other_exp_name, 'data_sample_name': sample_name, 'data_source_name': data_file_name.replace('.data', '')}
    else:
        tested_sample_config = {'exp_name': exp_name, 'data_sample_name': sample_name, 'data_source_name': sample_name}
    
    tested_sample_name = tested_sample_config['data_sample_name']
    tested_sample_data_source_name = tested_sample_config['data_source_name']
    tested_sample_from_exp = tested_sample_config['exp_name']
    
    task = sample_configs.get(tested_sample_name)
    if task != None:
        t = task[int(task_number) - 1]
        query = t['query']
        rg = t['range']
        logging.basicConfig(filename=f"{tested_sample_name}_{rg}.log", encoding='utf-8', level=logging.INFO)
        try:
            compute_and_save(
                exp_dir_path, tested_sample_name, query, rg,
                tested_sample_from_exp, tested_sample_data_source_name)
        except FileNotFoundError as e:
            logging.error(e)
    else:
        logging.warning(f'**no config for tested sample {tested_sample_name}')
            
            
if __name__ == '__main__':
    if len(sys.argv) > 1:
        exp_name = sys.argv[1]
        sample_name = sys.argv[2]
        task_number = sys.argv[3]
        cpu_number = sys.argv[4]
        exp_dir_path = os.path.join(data_dir, exp_name)
        if os.path.isdir(exp_dir_path):
            get_anchor_metrics(exp_dir_path, sample_name, task_number, cpu_number)
        else:
            print(f'**no exp dir {exp_dir_path}')