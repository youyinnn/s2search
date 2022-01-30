from s2search.rank import S2Ranker
import time, os, os.path as path, sys, yaml, json
from multiprocessing import Pool
import numpy as np
import feature_masking as fm

model_dir = './s2search_data'
data_dir = './pipelining'
ranker = None

def init_ranker():
    global ranker 
    if ranker == None:
        print(f'Loading ranker model...')
        st = time.time()
        ranker = S2Ranker(model_dir)
        et = round(time.time() - st, 2)
        print(f'Load the s2 ranker within {et} sec')


def get_scores(query, paper, mask_option='origin'):
    init_ranker()
    st = time.time()
    scores = ranker.score(query, paper)
    et = round(time.time() - st, 6)
    # print(paper)
    # print(f'Scores\n{scores} on option: {mask_option}')
    print(f'Compute {len(scores)} scores within {et} sec by masking option {mask_option}')
    return scores

def read_conf(exp_dir_path_str):
    conf_path = path.join(exp_dir_path_str, 'conf.yml')
    with open(str(conf_path), 'r') as f:
        conf = yaml.safe_load(f)
        return conf.get('description'), conf.get('samples'), 

def get_scores_and_save(arg):
    query = arg[0]
    paper_data = arg[1]
    exp_dir_path_str = arg[2]
    npy_file_name = arg[3]
    mask_option = arg[4]

    original_score_npy_file_name = path.join(exp_dir_path_str, 'scores', npy_file_name)
    if not os.path.exists(str(original_score_npy_file_name) + '.npy'):
        scores = get_scores(query, paper_data, mask_option)
        np.save(original_score_npy_file_name, scores)
    else:
        print(f'Scores of {original_score_npy_file_name} exist, should pass')

def score_file_is_configured(sample_configs, score_file_name):
    score_file_name = score_file_name.replace('.npy', '')
    exp_name, sample_data_name, task_name, one_masking_options = score_file_name.split('_')

    sample_tasks = sample_configs.get(sample_data_name)
    if sample_tasks != None:
        task_number = int(task_name[1:])
        if len(task_name) >= task_number:
            task = sample_tasks[task_number - 1]
            masking_option_keys = task['masking_option_keys']
            if one_masking_options == 'origin' or one_masking_options in masking_option_keys:
                return True

    return False

if __name__ == '__main__':
    if len(sys.argv) > 1:
        exp_list = sys.argv[1:]
        for exp_name in exp_list:
            exp_dir_path = path.join(data_dir, exp_name)
            exp_dir_path_str = str(exp_dir_path)
            if path.isdir(exp_dir_path):
                description, sample_configs = read_conf(exp_dir_path_str)
                print(f'\nRunning s2search ranker on {exp_name} experiment data')
                print(f'Description of this experiment: {description}')

                # scores dir
                scores_dir = path.join(exp_dir_path_str, 'scores')
                if not path.exists(str(scores_dir)):
                    os.mkdir(str(scores_dir))
                else:
                    for root, dirs, files in os.walk(scores_dir):
                        for file_name in files:
                            # remove score file if it is not configured
                            if not score_file_is_configured(sample_configs, file_name):
                                os.remove(path.join(exp_dir_path_str, 'scores', file_name))
                
                sample_file_list = [f for f in os.listdir(exp_dir_path_str) if path.isfile(path.join(exp_dir_path_str, f)) and f.endswith('.data')]

                for file_name in sample_file_list:   
                    paper_data = []
                    sample_name = file_name.replace('.data', '')
                    # # read sample data
                    with open(str(path.join(data_dir, exp_name, file_name))) as f:
                        lines = f.readlines()
                        for line in lines:
                            paper_data.append(json.loads(line.strip(), strict=False))

                    sample_task_list = sample_configs[sample_name]

                    t_count = 0
                    for task in sample_task_list:
                        t_count += 1
                        sample_query = task['query']
                        sample_masking_option_keys = task['masking_option_keys']
                        get_scores_and_save([sample_query, paper_data, exp_dir_path_str, f'{exp_name}_{sample_name}_t{t_count}_origin', 'origin'])

                        # masking
                        rs = fm.masking(paper_data, sample_masking_option_keys)
                        for key in sample_masking_option_keys:
                            get_scores_and_save([
                                sample_query,
                                rs[key],
                                exp_dir_path_str,
                                f'{exp_name}_{sample_name}_t{t_count}_{key}',
                                key
                            ])

                    print(f'Done with {exp_name} {file_name}')
                print(f'Done with {exp_name}')
                
            else:
                print(f'\nNo such dir: {str(exp_dir_path)}')
    else:
        print(f'Please provide the name of the experiment data folder.')

