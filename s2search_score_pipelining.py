from s2search.rank import S2Ranker
import time, os, os.path as path, sys, yaml, json, multiprocessing
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
        return conf['query'], conf.get('description'), conf.get('masking_option_keys'), conf.get('sample_name')

def get_scores_and_save(arg):
    query = arg[0]
    paper_data = arg[1]
    exp_dir_path_str = arg[2]
    npy_file_name = arg[3]
    mask_option = arg[4]

    scores = get_scores(query, paper_data, mask_option)
    original_score_npy_file_name = path.join(exp_dir_path_str, 'scores', npy_file_name)
    np.save(original_score_npy_file_name, scores)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        exp_list = sys.argv[1:]
        for exp_name in exp_list:
            exp_dir_path = path.join(data_dir, exp_name)
            exp_dir_path_str = str(exp_dir_path)
            if path.isdir(exp_dir_path):
                query, description, masking_option_keys, sample_name = read_conf(exp_dir_path_str)
                print(f'\nRunning s2search ranker on {exp_name} experiment data')
                print(f'Description of this experiment: {description}')

                scores_dir = path.join(exp_dir_path_str, 'scores')
                if not path.exists(str(scores_dir)):
                    os.mkdir(str(scores_dir))
                else:
                    for root, dirs, files in os.walk(scores_dir):
                        for file_name in files:
                            os.remove(path.join(exp_dir_path_str, 'scores', file_name))
                
                paper_data = []
                for sn in sample_name:
                    file_name = f'{sn}.data'
                    sample_file_name = path.join(exp_dir_path_str, file_name)
                    if path.isfile(sample_file_name):
                        print(f'Start computing {exp_name} {file_name}')
                    else:
                        print(f'No sample data {file_name} under {exp_name}')
                        continue
                
                    # read sample data
                    with open(str(path.join(data_dir, exp_name, file_name))) as f:
                        lines = f.readlines()
                        for line in lines:
                            paper_data.append(json.loads(line.strip(), strict=False))

                    sample_name = file_name.replace('.data', '')

                    get_scores_and_save([query, paper_data, exp_dir_path_str, f'{exp_name}_{sample_name}_origin', 'origin'])

                    # masking
                    rs = fm.masking(paper_data, masking_option_keys)
                    for key in masking_option_keys:
                        get_scores_and_save([
                            query,
                            rs[key],
                            exp_dir_path_str,
                            f'{exp_name}_{sample_name}_{key}',
                            key
                        ])

                    # clear the data for next sample file
                    paper_data = []
                    print(f'Done with {exp_name} {file_name}')
                print(f'Done with {exp_name}')
                
            else:
                print(f'\nNo such dir: {str(exp_dir_path)}')
    else:
        print(f'Please provide the name of the experiment data folder.')

