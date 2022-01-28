from os import listdir, path
import numpy as np, yaml

def ask(in_colab):
    # listing experiment dir
    if in_colab:
        data_dir = './s2search/pipelining'
    else:
        data_dir = './pipelining'

    experiments_dir = [dir for dir in listdir(data_dir) if path.isdir(path.join(data_dir, dir))]
    print(f'Got experiments: {experiments_dir}')

    # select exp
    exp = input("Typing the experiment name: ")
    exp_dir = path.join(data_dir, exp)
    conf_path = path.join(exp_dir, 'conf.yml')
    with open(conf_path, 'r') as f:
        conf = yaml.safe_load(f)
        # query = conf['query']; 
        description = conf.get('description')
        samples_config = conf.get('samples')

    print(f'Experiment {exp}\'s description: {description}')

    # listing exp samples
    sample_file_list = [f for f in listdir(exp_dir) if path.isfile(path.join(exp_dir, f)) and f.endswith('.data')]
    print(f'Got sample data: {sample_file_list}')
    
    # preparing data
    score_dir = path.join(exp_dir, 'scores')

    sample_and_data = []
    for file_name in sample_file_list:
        sample_name = file_name.replace('.data', '')

        sample_task_list = samples_config[sample_name]

        t_count = 0
        for task in sample_task_list:
            t_count += 1
            sample_query = task['query']
            sample_masking_option_keys = task['masking_option_keys']

            sample_origin_npy = np.load(path.join(score_dir, f'{exp}_{sample_name}_t{t_count}_origin.npy'))

            sample_feature_masking_npy = []
            for key in sample_masking_option_keys:
                sample_feature_masking_npy.append(np.load(path.join(score_dir, f'{exp}_{sample_name}_t{t_count}_{key}.npy')))
            feature_stack = np.stack((sample_feature_masking_npy))

            d_features = []
            for array in feature_stack:
                d_features.append(np.absolute((sample_origin_npy - array)))
            d_features = np.array(d_features)

            sample_and_data.append({
                'sample_and_task_name': f'{sample_name}-task{t_count}',
                'origin': sample_origin_npy,
                'd_features': d_features,
                'query': sample_query,
                'masking_option_keys': sample_masking_option_keys
            })

    return [
        exp, exp_dir, description,
        sample_and_data,
    ]