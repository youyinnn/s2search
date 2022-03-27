import os
import sys
import time
import numpy as np
from getting_data import get_categorical_encoded_data, decode_paper
from ranker_helper import get_scores, start_record_paper_count, end_record_paper_count, processing_log, get_current_paper_count
from s2search_score_pdp import save_pdp_to_npz
import pytz
import datetime
import logging
from sklearn.model_selection import train_test_split
import shap
import psutil

utc_tz = pytz.timezone('America/Montreal')

def get_time_str():
    return datetime.datetime.now(tz=utc_tz).strftime("%m/%d/%Y, %H:%M:%S")

def compute_and_save(output_exp_dir, output_data_sample_name, query, rg, data_exp_name, data_sample_name, logger):
    
    task_name = f'get categorical paper data for {output_exp_dir} {output_data_sample_name}'
    start_record_paper_count(task_name)
    categorical_name, paper_data = get_categorical_encoded_data(data_exp_name, data_sample_name, query)
    end_record_paper_count(task_name)
    
    def pred(X):
        paper_data = [
            decode_paper(categorical_name, x)
            for x in X
        ]
        return get_scores(query, paper_data, ptf= False)

    X_train, X_test = train_test_split(paper_data, test_size=0.2, random_state=0)

    task_name = f'sampling shap explainer training for {output_exp_dir} {output_data_sample_name}'
    start_record_paper_count(task_name)
    explainer = shap.SamplingExplainer(pred, X_train)
    end_record_paper_count(task_name)
    
    task_name = f'sampling shap sv computing for {output_exp_dir} {output_data_sample_name}'
    start_record_paper_count(task_name)
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
    logger.info(f'\n[{get_time_str()}] start computing sampling shap for range: {rg}')
    rate = 10
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
        logger.info(f'[{get_time_str()}] computing sampling shap {curr_start}:{curr_end} within {round(time.time() - stt, 4)} sec, {avgt} on average')
        logger.info(f'[{get_time_str()}] instances left: {end-start-count}')
        logger.info(f'[{get_time_str()}] estimating time left: {datetime.timedelta(seconds=((end-start-count) * avgt))}')
        logger.info(f'\n{shap_v.values}')
        processing_log(f'instances left: {end-start-count}')
        processing_log(f'estimating time left: {datetime.timedelta(seconds=((end-start-count) * avgt))}')
        processing_log(f'---- {count}:{get_current_paper_count()}')
        
        save_pdp_to_npz('.', metrics_npz_file, 
            shap_values=shap_values,
            base_values=list(base_values),
        )
        
        curr_start = curr_end
        curr_end += rate
    
    end_record_paper_count(task_name)
    logger.info(f'[{get_time_str()}] finish computing sampling shap for range: {rg} within {round(time.time() - st, 6)} sec')
    
def get_sampling_shap_shapley_value(exp_dir_path, query, task_config, data_info):
    current_sample_name = data_info['current_sample_name']
    sample_src_name = data_info['sample_src_name']
    sample_src_exp_name = data_info['sample_src_exp_name']
    rg = task_config['range']
    
    log_dir = os.path.join(exp_dir_path, 'log')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_file_path = os.path.join(log_dir, f'{current_sample_name}_shap_{rg}.log')
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(filename=log_file_path, encoding='utf-8'))  
    
    if sys.platform != "darwin":
        p = psutil.Process()
        worker = task_config['cpu']
        logger.info(f"Child #{worker}: {p}, affinity {p.cpu_affinity()}")
        p.cpu_affinity(worker)
        logger.info(f"Child #{worker}: Set my affinity to {worker}, affinity now {p.cpu_affinity()}")
    
    compute_and_save(
        exp_dir_path, current_sample_name, query, rg,
        sample_src_exp_name, sample_src_name, logger)
    
# def get_shap_metrics(exp_dir_path, sample_name, task_number, cpu_number):

#     if sys.platform != "darwin":
#         p = psutil.Process()
#         worker = int(cpu_number)
#         print(f"Child #{worker}: {p}, affinity {p.cpu_affinity()}", flush=True)
#         p.cpu_affinity([worker])
#         print(f"Child #{worker}: Set my affinity to {worker}, affinity now {p.cpu_affinity()}", flush=True)

#     des, sample_configs, sample_from_other_exp = read_conf(exp_dir_path)
    
#     if sample_name in sample_from_other_exp.keys():
#         other_exp_name, data_file_name = sample_from_other_exp.get(sample_name)
#         tested_sample_config = {'exp_name': other_exp_name, 'data_sample_name': sample_name, 'data_source_name': data_file_name.replace('.data', '')}
#     else:
#         tested_sample_config = {'exp_name': exp_name, 'data_sample_name': sample_name, 'data_source_name': sample_name}
    
#     tested_sample_name = tested_sample_config['data_sample_name']
#     tested_sample_data_source_name = tested_sample_config['data_source_name']
#     tested_sample_from_exp = tested_sample_config['exp_name']
    
#     task = sample_configs.get(tested_sample_name)
#     if task != None:
#         t = task[int(task_number) - 1]
#         query = t['query']
#         rg = t['range']
#         logging.basicConfig(filename=f"{tested_sample_name}_shap_{rg}.log", encoding='utf-8', level=logging.INFO)
#         try:
#             compute_and_save(
#                 exp_dir_path, tested_sample_name, query, rg,
#                 tested_sample_from_exp, tested_sample_data_source_name)
#         except FileNotFoundError as e:
#             logging.error(e)
#     else:
#         logging.warning(f'**no config for tested sample {tested_sample_name}')
            
            
# if __name__ == '__main__':
#     if len(sys.argv) > 1:
#         exp_name = sys.argv[1]
#         sample_name = sys.argv[2]
#         task_number = sys.argv[3]
#         cpu_number = sys.argv[4]
#         exp_dir_path = os.path.join(data_dir, exp_name)
#         if os.path.isdir(exp_dir_path):
#             get_shap_metrics(exp_dir_path, sample_name, task_number, cpu_number)
#         else:
#             print(f'**no exp dir {exp_dir_path}')