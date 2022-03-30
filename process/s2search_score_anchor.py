import time
import os
import sys
import numpy as np
from getting_data import load_sample, feature_key_list, get_categorical_encoded_data, decode_paper
from ranker_helper import get_scores, start_record_paper_count, end_record_paper_count, processing_log
from s2search_score_pdp import save_pdp_to_npz
from anchor import anchor_tabular
import pytz
import datetime
import logging
import psutil

utc_tz = pytz.timezone('America/Montreal')


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


def compute_and_save(output_exp_dir, output_data_sample_name, query, rg, data_exp_name, data_sample_name, logger, explainer_configs):
    metrics_npz_file = os.path.join(
        output_exp_dir, 'scores', f'{output_data_sample_name}_anchor_metrics_{rg[0]}_{rg[1]}.npz')

    st = time.time()
    logger.info(f'\n[{get_time_str()}] start anchor metrics')
    paper_data = load_sample(data_exp_name, data_sample_name, not_df=True)

    task_name = f'get prediction of paper data for {output_exp_dir} {output_data_sample_name} {rg}'
    start_record_paper_count(task_name)
    y_pred_file = os.path.join(
        output_exp_dir, 'scores', f'{data_sample_name}_y_pred.npz')
    if not os.path.exists(y_pred_file):
        y_pred = get_scores(query, paper_data)
        save_pdp_to_npz('.', y_pred_file, y_pred)
    else:
        y_pred = np.load(y_pred_file)['arr_0']
    end_record_paper_count(task_name)

    # make class_name
    class_name = ['(,-17]', '(-17, -10]', '(-10, -5]', '(-5, <0]',
                  '(0, 3]', '(3, 5]', '(5, 6]', '(6, 7]', '(7, 8]', '(8, 9]', '(9,)']

    task_name = f'get categorical paper data for {output_exp_dir} {output_data_sample_name} {rg}'
    start_record_paper_count(task_name)
    categorical_name, paper_data = get_categorical_encoded_data(
        data_exp_name, data_sample_name, query, paper_data)
    end_record_paper_count(task_name)

    explainer = anchor_tabular.AnchorTabularExplainer(
        class_name,
        feature_key_list,
        paper_data,
        categorical_name)

    def pred_fn(x):
        predictions = get_scores(query, [decode_paper(
            categorical_name, p) for p in x], ptf=False)
        encoded_pred = [class_name.index(get_class(pp)) for pp in predictions]
        return np.array(encoded_pred)

    data_len = len(paper_data)
    numerator, denominator = rg
    process_len = int(data_len / denominator) + 1
    curr_numerator = 1
    start = 0
    end = process_len
    while curr_numerator != numerator:
        start += process_len
        end = end + process_len if end + process_len < data_len else data_len
        curr_numerator += 1

    metrics = dict(
        title=[],
        abstract=[],
        venue=[],
        authors=[],
        year=[],
        n_citations=[],
    )

    previous_work_idx = start
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

    task_name = f'get anchor metrics for {output_exp_dir} {output_data_sample_name} {rg} from index: {previous_work_idx} to {end - 1}'
    start_record_paper_count(task_name)

    sst = time.time()
    th = explainer_configs.get('threshold') if explainer_configs.get(
        'threshold') != None else 0.9999
    tau = explainer_configs.get(
        'tau') if explainer_configs.get('tau') != None else 0.2
    logger.info(
        f'[{get_time_str()}] start computing anchor from index: {previous_work_idx} to {end - 1} with config: {th} {tau}')
    count = 0
    for i in range(previous_work_idx, end):
        exp = explainer.explain_instance(
            paper_data[i], pred_fn, threshold=th, tau=tau)

        previous_single_precision = 0
        for j in range(len(exp.names())):
            name = exp.names()[j]
            current_single_precision = exp.precision(
                j) - previous_single_precision
            previous_single_precision = exp.precision(j)
            for feature_name in metrics.keys():
                if name.startswith(f'{feature_name}'):
                    metrics[feature_name].append(current_single_precision)

        count += 1

        if count % 10 == 0:
            ett = round(time.time() - sst, 6)
            avg_time = round(ett / count, 4)
            logger.info(f'[{get_time_str()}] ({i} / {end - 1}) {metrics_to_str(metrics)} \
within ({ett} total / {avg_time} average)')
            logger.info(
                f'estimate time left: {datetime.timedelta(seconds=((end-previous_work_idx-count) * avg_time))}')
            save_pdp_to_npz('.', metrics_npz_file,
                            title=metrics['title'],
                            abstract=metrics['abstract'],
                            venue=metrics['venue'],
                            authors=metrics['authors'],
                            year=metrics['year'],
                            n_citations=metrics['n_citations'],
                            idx=[i]
                            )

    end_record_paper_count(task_name)
    logger.info(
        f'[{get_time_str()}] end anchor metrics witin {round(time.time() - st, 6)}')


def get_anchor_metrics(exp_dir_path, query, task_config, explainer_configs, data_info):
    current_sample_name = data_info['current_sample_name']
    sample_src_name = data_info['sample_src_name']
    sample_src_exp_name = data_info['sample_src_exp_name']
    rg = task_config['range']

    log_dir = os.path.join(exp_dir_path, 'log')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_file_path = os.path.join(
        log_dir, f'{current_sample_name}_anchor_{rg}.log')

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # remain one log file handler
    if logger.hasHandlers():
        for h in logger.handlers:
            logger.removeHandler(h)
    logger.addHandler(logging.FileHandler(
        filename=log_file_path, encoding='utf-8'))

    rg = task_config['range']

    if sys.platform != "darwin":
        p = psutil.Process()
        worker = task_config['cpu']
        logger.info(f"\nChild #{worker}: {p}, affinity {p.cpu_affinity()}")
        p.cpu_affinity(worker)
        logger.info(
            f"Child #{worker}: Set my affinity to {worker}, affinity now {p.cpu_affinity()}")

    compute_and_save(
        exp_dir_path, current_sample_name, query, rg,
        sample_src_exp_name, sample_src_name, logger, explainer_configs)
