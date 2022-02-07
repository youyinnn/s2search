from os import path
import numpy as np, yaml

def get(exp_name, sample_name):
    exp_dir = '.'
    conf_path = path.join(exp_dir, 'conf.yml')
    with open(conf_path, 'r') as f:
        conf = yaml.safe_load(f)
        samples_config = conf.get('samples')

    # sample_file_list = samples_config.keys()
    print(f'Got sample data: {sample_name}')

    # preparing data
    score_dir = path.join(exp_dir, 'scores')

    sample_data_and_config = []
    sample_task_list = samples_config[sample_name]

    t_count = 0
    for task in sample_task_list:
        t_count += 1
        sample_query = task['query']
        sample_masking_option_keys = task['masking_option_keys']

        sample_origin_npy = np.load(path.join(score_dir, f'{exp_name}_{sample_name}_t{t_count}_origin.npz'))['arr_0']

        sample_feature_masking_npy = []
        for key in sample_masking_option_keys:
            sample_feature_masking_npy.append(np.load(path.join(score_dir, f'{exp_name}_{sample_name}_t{t_count}_{key}.npz'))['arr_0'])
        if len(sample_feature_masking_npy) > 0:
            feature_stack = np.stack((sample_feature_masking_npy))
        else:
            feature_stack = np.stack(([sample_origin_npy]))
            sample_masking_option_keys.append('_origin')

        sample_data_and_config.append({
            'sample_and_task_name': f'{sample_name}',
            'task_number': t_count,
            'query': sample_query,
            'origin': sample_origin_npy,
            'feature_stack': feature_stack,
            'masking_option_keys': sample_masking_option_keys
        })

    return sample_data_and_config