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
        masking_option_keys = conf['masking_option_keys']
        sample_name = conf['sample_name']

    print(f'Experiment {exp}\'s description: {description}')

    # listing exp samples
    print(f'Got sample data: {sample_name}')

    # select sample
    sample = input("Typing the sample name: ")

    # preparing data
    score_dir = path.join(exp_dir, 'scores')
    sample_origin_npy = np.load(path.join(score_dir, f'{exp}_{sample}_origin.npy'))

    sample_feature_masking_npy = []

    for key in masking_option_keys:
        sample_feature_masking_npy.append(np.load(path.join(score_dir, f'{exp}_{sample}_{key}.npy')))

    feature_stack = np.stack((sample_feature_masking_npy))

    d_features = []
    for array in feature_stack:
        d_features.append(np.absolute((sample_origin_npy-array)))

    d_features = np.array(d_features)

    return [
        masking_option_keys, 
        sample_name, sample_origin_npy, d_features,
        exp, description
    ]