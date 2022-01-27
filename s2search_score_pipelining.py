from s2search.rank import S2Ranker
import time, os, os.path as path, sys, yaml, json
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
    # st = time.time()
    scores = ranker.score(query, paper)
    # et = round(time.time() - st, 6)
    # print(paper)
    # print(f'Scores\n{scores}')
    # print(f'Compute {len(scores)} scores by masking option {mask_option} within {et} sec')
    return scores

def read_conf(exp_dir_path_str):
    conf_path = path.join(exp_dir_path_str, 'conf.yml')
    with open(str(conf_path), 'r') as f:
        conf = yaml.safe_load(f)
        return conf['query'], conf.get('description')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        exp_name = sys.argv[1]
        exp_dir_path = path.join(data_dir, exp_name)
        exp_dir_path_str = str(exp_dir_path)
        if path.isdir(exp_dir_path):
            query, description = read_conf(exp_dir_path_str)
            print(f'Running s2search ranker on {exp_name} experiment data.')
            print(f'Description of this experiment: {description}')

            for root, dirs, files in os.walk(path.join(exp_dir_path_str, 'scores')):
                for file_name in files:
                    os.remove(path.join(exp_dir_path_str, 'scores', file_name))
            
            paper_data = []
            for root, dirs, files in os.walk(exp_dir_path_str):
                for file_name in files:
                    if file_name.endswith((".data")):
                        with open(str(path.join(data_dir, exp_name, file_name))) as f:
                            lines = f.readlines()
                            for line in lines:
                                paper_data.append(json.loads(line.strip(), strict=False))

                        scores = get_scores(query, paper_data)
                        sample_name = file_name.replace('.data', '')
                        original_score_npy_file_name = path.join(exp_dir_path_str, 'scores', f'{exp_name}_{sample_name}_origin')
                        np.save(original_score_npy_file_name, scores)

                        # masking
                        rs = fm.masking(paper_data)
                        for key in fm.masking_options.keys():
                            masking_score = get_scores(query, rs[key], key)
                            masking_score_npy_file_name = path.join(exp_dir_path_str, 'scores', f'{exp_name}_{sample_name}_{key}')
                            np.save(masking_score_npy_file_name, masking_score)

                        # clear the data for next sample file
                        paper_data = []
        else:
            print(f'No such dir: {str(exp_dir_path)}')
    else:
        print(f'Please provide the name of the experiment data folder.')

