from s2search.rank import S2Ranker
import time
import os
import sys
import os.path as path
import psutil
from functools import reduce
import math
import pytz
import datetime
from multiprocessing import Pool
import numpy as np
import psutil
from pathlib import Path as pt
import logging

utc_tz = pytz.timezone('America/Montreal')

mem = psutil.virtual_memory()
zj = float(mem.total) / 1024 / 1024 / 1024
work_load = 1 if math.ceil(zj / 16) == 1 else math.ceil(zj / 16)
if os.environ.get('S2_MODEL_WORKLOAD') != None:
    print('using env workload')
    work_load = int(os.environ.get('S2_MODEL_WORKLOAD'))

gb_ranker = []
gb_ranker_enable = False

paper_count = 0
recording_paper_count = False

ranker_logger = None

def set_ranker_logger(exp_dir_path, method = None):
    method = method if method != None else 'unknown'
    log_dir = os.path.join(exp_dir_path, 'log')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_file_path = os.path.join(log_dir, f'ranker_calls_{method}_{datetime.datetime.now(tz=utc_tz).strftime("%m-%d-%Y-%H-%M-%S")}.log')
    global ranker_logger
    ranker_logger = logging.getLogger(__name__)
    ranker_logger.setLevel(logging.INFO)
    ranker_logger.addHandler(logging.FileHandler(filename=log_file_path, encoding='utf-8'))
    
def log_info(task_name, msg):
    if task_name != None and ranker_logger != None:
        ranker_logger.info(f'[{get_time_str()}] [{task_name}]\n{msg}')
        
def processing_log(msg):
    global ranker_logger
    ranker_logger.info(msg)
    
def get_current_paper_count():
    global paper_count
    return paper_count

def start_record_paper_count(task_name):
    global recording_paper_count, ranker_logger
    recording_paper_count = True
    ranker_logger.info('\n')
    log_info(task_name, f'============== start ==============')
    
def end_record_paper_count(task_name):
    global recording_paper_count, paper_count, ranker_logger
    current_number = paper_count
    recording_paper_count = False
    paper_count = 0
    log_info(task_name, f'============== end {current_number} ==============')
    return current_number

def get_time_str():
    return datetime.datetime.now(tz=utc_tz).strftime("%m/%d/%Y, %H:%M:%S")

def enable_global(ptf = False):
    global gb_ranker_enable
    gb_ranker_enable = True
    global gb_ranker
    if len(gb_ranker) == 0:
        gb_ranker.append(init_ranker(ptf))
        
def disable_global():
    global gb_ranker_enable
    gb_ranker_enable = False
    global gb_ranker
    gb_ranker = []

def check_model_existance(default_dir = path.join(os.getcwd(), 's2search_data')):
    while not default_dir.endswith('/s2search'):
        default_dir = path.join(pt(default_dir).parents[0])

    if default_dir.endswith('/s2search'):
        default_dir = path.join(default_dir, 's2search_data')

    if os.path.exists(default_dir):
        list_files = [f for f in os.listdir(default_dir) if os.path.isfile(os.path.join(default_dir, f))]
        if 'titles_abstracts_lm.binary' in list_files \
            and 'authors_lm.binary' in list_files \
                and 'lightgbm_model.pickle' in list_files \
                    and 'venues_lm.binary' in list_files:
            return default_dir
    else:
        return os.environ.get('S2_MODEL_DATA')

def init_ranker(ptf = False):
    data_dir = check_model_existance()
    if not ptf:
        st = time.time()
        print(f'Loading process ranker model...')
    ranker = S2Ranker(data_dir)
    if not ptf:
        et = round(time.time() - st, 2)
        print(f'Load the process s2 ranker within {et} sec')
    return ranker

def get_ranker(ptf = False):
    global gb_ranker_enable
    global gb_ranker
    if ptf:
        print(f"get ranker in {os.getpid()} with global setting: {gb_ranker_enable} and gb_ranker len {len(gb_ranker)}")
    if gb_ranker_enable:
        return gb_ranker[0]
    else:
        return init_ranker(ptf)

def find_weird_score(scores, paper_list):
    weird_paper_idx = []
    weird_paper = []
    for i in range(len(scores)):
        score = scores[i]
        if score > 100:
            weird_paper_idx.append(i)
            weird_paper.append(paper_list[i])
            
    return weird_paper_idx, weird_paper
    

def get_scores(query, paper, task_name=None, ptf=True, force_global = False):    
    log_info(task_name, f'get scores for {len(paper)} papers')
    global recording_paper_count, paper_count
    if recording_paper_count:
        paper_count += len(paper)
    st = time.time()
    ts = datetime.datetime.now(tz=utc_tz).strftime("%m/%d/%Y, %H:%M:%S")
    if work_load == 1 or force_global:
        if not force_global and ptf:
            print('fail to not force global because 1 worker available')
        enable_global(ptf)
        scores = get_scores_for_one_worker([query, paper, task_name, -1, ptf])
    else:
        disable_global()
        paper_limit_for_a_worker = math.ceil(len(paper) / work_load)
        if ptf:
            print(f'[{ts}] with {work_load} workloads, porcessing {paper_limit_for_a_worker} papers per workload')
        task_arg = []
        curr_idx = 0
        idx = 0
        while curr_idx < len(paper):
            end_idx = curr_idx + paper_limit_for_a_worker if curr_idx + paper_limit_for_a_worker < len(paper) else len(paper)
            task_arg.append(
                [
                    query,
                    paper[curr_idx: end_idx],
                    task_name,
                    idx,
                    ptf,
                ]
            )
            curr_idx += paper_limit_for_a_worker
            idx += 1
        with Pool(processes=work_load) as worker:
            rs = worker.map_async(get_scores_for_one_worker, task_arg)
            scores = rs.get()
        
    et = round(time.time() - st, 6)
    ts = datetime.datetime.now(tz=utc_tz).strftime("%m/%d/%Y, %H:%M:%S")
    log_info(task_name, f'task end within {et} sec')
    if ptf:
        print(f"[{'Main taks' if task_name == None else task_name}][{ts}] {len(paper)} scores within {et} sec ")
    if len(scores) == 1:
        return np.array(scores)
    return reduce(lambda x,y: np.append(x, y), scores)

def get_scores_for_one_worker(pos_arg):
    query, paper, task_name, task_number, ptf = pos_arg
    
    if sys.platform != "darwin" and task_number > -1:
        p = psutil.Process()
        worker = int(task_number)
        p.cpu_affinity([worker])
    
    one_ranker = get_ranker(ptf)
    if ptf:
        print(f"[{'Main taks' if task_name == None else task_name}:{task_number}] compute {len(paper)} scores with worker {os.getpid()}")
    scores = []
    paper_list = paper
    if len(paper_list) > 1000:
        curr_idx = 0
        while curr_idx < len(paper_list):
            end_idx = curr_idx + 1000 if curr_idx + 1000 < len(paper_list) else len(paper_list)
            curr_list = paper_list[curr_idx: end_idx]
            scores.extend(one_ranker.score(query, curr_list))
            curr_idx += 1000
    else:
        scores = one_ranker.score(query, paper_list)
        
    weird_paper_idx, weird_paper = find_weird_score(scores, paper_list)
            
    if len(weird_paper) > 0:
        fixed_score = [one_ranker.score(query, [one_paper])[0] for one_paper in weird_paper]
        idx = 0
        for weird_idx in weird_paper_idx:
            scores[weird_idx] = fixed_score[idx]
            idx += 0
    
    weird_paper_idx_again, _ = find_weird_score(scores, paper_list)

    if len(weird_paper_idx_again) > 0:
        print(f'still got weird scores')

    return scores
