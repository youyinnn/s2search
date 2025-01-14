from s2search.rank import S2Ranker
import time
import os
import os.path as path
import sys
import yaml
import json
import shutil
import numpy as np
import feature_masking as fm
from ranker_helper import get_scores, start_record_paper_count, end_record_paper_count

data_dir = str(path.join(os.getcwd(), 'pipelining'))
data_loading_line_limit = 100000


def read_conf(exp_dir_path_str):
    conf_path = path.join(exp_dir_path_str, 'conf.yml')
    with open(str(conf_path), 'r') as f:
        conf = yaml.safe_load(f)
        return conf.get('description'), conf.get('samples'), conf.get('sample_from_other_exp'),


def get_scores_and_save(arg):
    query = arg[0]
    paper_data = arg[1]
    exp_dir_path_str = arg[2]
    npy_file_name = arg[3]
    mask_option = arg[4]

    original_score_npy_file_name = path.join(
        exp_dir_path_str, 'scores', npy_file_name)
    scores = get_scores(query, paper_data, ptf=False,
                        task_name=f'masking_{npy_file_name}')
    incomplete_file = str(original_score_npy_file_name) + '#incomplete.txt'

    scores = [str(score) for score in scores]
    with open(incomplete_file, "a+") as f:
        f.write('\n'.join(scores) + '\n')


def score_file_is_configured(sample_configs, score_file_name):
    score_file_name = score_file_name.replace('.npz', '')
    score_file_name = score_file_name.replace('#incomplete.txt', '')
    exp_name, sample_data_name, task_name, one_masking_options = score_file_name.split(
        '_')

    sample_tasks = sample_configs.get(sample_data_name)
    if sample_tasks != None:
        task_number = int(task_name[1:])
        if len(sample_tasks) >= task_number:
            task = sample_tasks[task_number - 1]
            masking_option_keys = task['masking_option_keys']
            if one_masking_options == 'origin' or one_masking_options in masking_option_keys:
                return True

    return False


def txt_to_npy(exp_dir, exp_name, sample_name, task_name, masking_option_key, et):
    incomplete_file = path.join(
        exp_dir, 'scores', f'{exp_name}_{sample_name}_{task_name}_{masking_option_key}#incomplete.txt')
    arr = np.loadtxt(incomplete_file)
    complete_file = path.join(
        exp_dir, 'scores', f'{exp_name}_{sample_name}_{task_name}_{masking_option_key}')
    np.savez_compressed(complete_file, arr)
    os.remove(incomplete_file)
    print(
        f'Score computing for {exp_name}_{sample_name}_{task_name}_{masking_option_key} is done within {et} sec.')


def masking_score(pipelining_dir, exp_name, data_info, sample_config):

    current_sample_name = data_info['current_sample_name']
    sample_src_name = data_info['sample_src_name']
    sample_src_exp_name = data_info['sample_src_exp_name']

    exp_dir_path_str = str(os.path.join(pipelining_dir, exp_name))

    t_count = 0
    for task in sample_config:
        t_count += 1
        sample_query = task['query']
        sample_masking_option_keys = task['masking_option_keys']
        data_file_path = path.join(
            pipelining_dir, exp_name, f'{current_sample_name}.data')
        if not path.exists(data_file_path):
            data_file_path = path.join(
                pipelining_dir, sample_src_exp_name, f'{sample_src_name}.data')
            print(
                f'Using {data_file_path} for {exp_name} {current_sample_name}')
        paper_data = []

        # computing for original
        using_origin_from = task.get('using_origin_from')
        target_origin_exist = path.join(
            exp_dir_path_str, 'scores', f'{exp_name}_{current_sample_name}_{using_origin_from}_origin.npz')
        original_npy_file = path.join(
            exp_dir_path_str, 'scores', f'{exp_name}_{current_sample_name}_t{t_count}_origin.npz')
        if using_origin_from != None and target_origin_exist:
            print(
                f'Using origin result of {using_origin_from} for {exp_name}_{current_sample_name}_t{t_count}_origin.npz')
            shutil.copyfile(target_origin_exist,
                            original_npy_file)
        else:
            if not os.path.exists(original_npy_file):
                incomplete_original_npy_file = path.join(
                    exp_dir_path_str, 'scores', f'{exp_name}_{current_sample_name}_t{t_count}_origin#incomplete.txt')
                if path.exists(incomplete_original_npy_file):
                    with open(incomplete_original_npy_file) as f:
                        previous_progress = len(f.readlines())
                    print(
                        f'continue computing: {original_npy_file}')
                else:
                    previous_progress = 0
                    print(
                        f'start computing: {original_npy_file}')

                with open(str(data_file_path)) as f:
                    start_record_paper_count(
                        f'{exp_name}_{current_sample_name}_t{t_count}_origin')
                    line_count = 0
                    idx = 0
                    st = time.time()
                    for line in f:
                        idx += 1
                        if (idx <= previous_progress):
                            continue
                        paper_data.append(json.loads(
                            line.strip(), strict=False))
                        line_count += 1
                        if (line_count == data_loading_line_limit):
                            get_scores_and_save([
                                sample_query, paper_data, exp_dir_path_str,
                                f'{exp_name}_{current_sample_name}_t{t_count}_origin',
                                'origin',
                            ])
                            paper_data = []
                            line_count = 0
                    if len(paper_data) > 0:
                        get_scores_and_save([
                            sample_query, paper_data, exp_dir_path_str,
                            f'{exp_name}_{current_sample_name}_t{t_count}_origin',
                            'origin',
                        ])
                    end_record_paper_count(
                        f'{exp_name}_{current_sample_name}_t{t_count}_origin')
                    txt_to_npy(exp_dir_path_str, exp_name,
                               current_sample_name, f't{t_count}', 'origin', round(time.time() - st, 6))
            else:
                print(
                    f'Scores of {exp_name}_{current_sample_name}_t{t_count}_origin.npz exist, should pass')

        # computing for masking
        for key in sample_masking_option_keys:
            paper_data = []
            feature_masked_npy_file = path.join(
                exp_dir_path_str, 'scores', f'{exp_name}_{current_sample_name}_t{t_count}_{key}.npz')
            if not os.path.exists(feature_masked_npy_file):
                incomplete_original_npy_file = path.join(
                    exp_dir_path_str, 'scores', f'{exp_name}_{current_sample_name}_t{t_count}_{key}#incomplete.txt')
                if path.exists(incomplete_original_npy_file):
                    with open(incomplete_original_npy_file) as f:
                        previous_progress = len(f.readlines())
                        print(
                            f'continue computing: {feature_masked_npy_file}')
                else:
                    previous_progress = 0
                    print(
                        f'start computing: {feature_masked_npy_file}')

                with open(str(data_file_path)) as f:
                    start_record_paper_count(
                        f'{exp_name}_{current_sample_name}_t{t_count}_{key}')
                    line_count = 0
                    idx = 0
                    st = time.time()
                    for line in f:
                        idx += 1
                        if (idx <= previous_progress):
                            continue
                        paper_data.append(json.loads(
                            line.strip(), strict=False))
                        line_count += 1
                        if (line_count == data_loading_line_limit):
                            paper_data = fm.masking_with_option(
                                paper_data, fm.masking_options[key])
                            get_scores_and_save([
                                sample_query,
                                paper_data,
                                exp_dir_path_str,
                                f'{exp_name}_{current_sample_name}_t{t_count}_{key}',
                                key,
                            ])
                            paper_data = []
                            line_count = 0
                    if len(paper_data) > 0:
                        paper_data = fm.masking_with_option(
                            paper_data, fm.masking_options[key])
                        get_scores_and_save([
                            sample_query,
                            paper_data,
                            exp_dir_path_str,
                            f'{exp_name}_{current_sample_name}_t{t_count}_{key}',
                            key,
                        ])
                    end_record_paper_count(
                        f'{exp_name}_{current_sample_name}_t{t_count}_{key}')
                    txt_to_npy(exp_dir_path_str, exp_name,
                               current_sample_name, f't{t_count}', key, round(time.time() - st, 6))
            else:
                print(
                    f'Scores of {exp_name}_{current_sample_name}_t{t_count}_{key}.npz exist, should pass')
    print(f'Done with {exp_name} {current_sample_name}')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        exp_list = sys.argv[1:]
        # for exp_name in exp_list:
        #     exp_dir_path = path.join(data_dir, exp_name)
        #     exp_dir_path_str = str(exp_dir_path)
        #     if path.isdir(exp_dir_path):
        #         description, sample_configs, sample_from_other_exp = read_conf(
        #             exp_dir_path_str)
        #         print(
        #             f'\nRunning s2search ranker on {exp_name} experiment data')
        #         print(f'Description of this experiment: {description}')

        #         # scores dir
        #         scores_dir = path.join(exp_dir_path_str, 'scores')
        #         if not path.exists(str(scores_dir)):
        #             os.mkdir(str(scores_dir))
        #         else:
        #             for root, dirs, files in os.walk(scores_dir):
        #                 for file_name in files:
        #                     # remove score file if it is not configured
        #                     if not score_file_is_configured(sample_configs, file_name):
        #                         os.remove(
        #                             path.join(exp_dir_path_str, 'scores', file_name))

        #         # sample_file_list = [f for f in os.listdir(exp_dir_path_str) if path.isfile(path.join(exp_dir_path_str, f)) and f.endswith('.data')]
        #         sample_file_list = sample_configs.keys()

        #         for file_name in sample_file_list:
        #             sample_name = file_name.replace('.data', '')
        #             sample_task_list = sample_configs[sample_name]

        #             masking_score(sample_name, sample_task_list)
        #         print(f'Done with {exp_name}')

        #     else:
        #         print(f'\nNo such dir: {str(exp_dir_path)}')
    else:
        print(f'Please provide the name of the experiment data folder.')
