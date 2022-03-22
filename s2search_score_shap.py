import os
import sys
import time
import numpy as np
from s2search_score_pipelining import read_conf
from getting_data import load_sample, feature_key_list, get_categorical_encoded_data, decode_paper
from ranker_helper import get_scores
from s2search_score_pdp import save_pdp_to_npz
import pytz
import datetime
import logging
from sklearn.model_selection import train_test_split
import shap

utc_tz = pytz.timezone('America/Montreal')

data_dir = str(os.path.join(os.getcwd(), 'pipelining'))

def get_time_str():
    return datetime.datetime.now(tz=utc_tz).strftime("%m/%d/%Y, %H:%M:%S")

def compute_and_save(output_exp_dir, output_data_sample_name, query, rg, data_exp_name, data_sample_name):
    
    categorical_name, paper_data = get_categorical_encoded_data(data_exp_name, data_sample_name, query)
    
    def pred(X):
        paper_data = [
            decode_paper(categorical_name, x)
            for x in X
        ]
        return get_scores('Machine Learning', paper_data, ptf= False)

    X_train, X_test = train_test_split(paper_data, test_size=0.2, random_state=0)

    explainer = shap.SamplingExplainer(pred, X_train)
    
    test_len = len(X_test)
    numerator, denominator = rg
    process_len = int(test_len / denominator) + 1
    curr_numerator = 1
    start = 0
    end = process_len
    while curr_numerator != numerator:
        start += process_len
        end = end + process_len if end + process_len < test_len else test_len
        curr_numerator += 1     
        
    metrics_npz_file = os.path.join(output_exp_dir, 'scores', f'{output_data_sample_name}_shap_sampling_{rg[0]}_{rg[1]}.npz')

    st = time.time()
    logging.info(f'\n[{get_time_str()}] start computing sampling shap for range: {rg}')
    rate = 2
    curr_start = start
    curr_end = start + rate
    shap_values = []
    base_values = set([])
    count = 0
    while curr_end < end:
        stt = time.time()
        shap_v = explainer(X_test[curr_start : curr_end])
        count += rate
        shap_values.extend(list(shap_v.values))
        base_values.add(shap_v.base_values)
        avgt = round((time.time() - st) / (count) ,4)
        logging.info(f'[{get_time_str()}] computing sampling shap {curr_start}:{curr_end} within {round(time.time() - stt, 4)} sec, {avgt} on average')
        logging.info(f'[{get_time_str()}] instances left: {end-start-count}')
        logging.info(f'[{get_time_str()}] estimating time left: {datetime.timedelta(seconds=((end-start-count) * avgt))}')
        logging.info(f'\n{shap_v.values}')
        
        save_pdp_to_npz('.', metrics_npz_file, 
            shap_values=shap_values,
            base_values=list(base_values),
        )
        
        curr_start = curr_end
        curr_end += rate
    
    logging.info(f'[{get_time_str()}] finish computing sampling shap for range: {rg} within {round(time.time() - st, 6)} sec')
    
    
def get_shap_metrics(exp_dir_path, sample_name, task_number):

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
        logging.basicConfig(filename=f"{tested_sample_name}_shap_{rg}.log", encoding='utf-8', level=logging.INFO)
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
        exp_dir_path = os.path.join(data_dir, exp_name)
        if os.path.isdir(exp_dir_path):
            get_shap_metrics(exp_dir_path, sample_name, task_number)
        else:
            print(f'**no exp dir {exp_dir_path}')