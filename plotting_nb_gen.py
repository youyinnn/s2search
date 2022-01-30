import os, os.path as path, yaml, sys
import nbformat as nbf

data_dir = './pipelining'
exp_dir_list = [d for d in os.listdir(data_dir) if path.isdir(path.join(data_dir, d))]

if __name__ == '__main__':
    if len(sys.argv) == 1:
        exp_list = exp_dir_list
        force = False
    if len(sys.argv) == 2:
        exp_list = [sys.argv[1]]
        force = True
    if len(sys.argv) == 3:
        exp_list = [sys.argv[1]]
        force = True

    for exp_name in exp_list:
        if exp_name not in exp_dir_list:
            print(f'No {exp_name} experiment')
            continue

        exp_dir = path.join(data_dir, exp_name)
        conf_path = path.join(exp_dir, 'conf.yml')
        with open(conf_path, 'r') as f:
            conf = yaml.safe_load(f)
            description = conf.get('description')
            sample_configs = conf.get('samples')
            sample_list = [sys.argv[2]] if len(sys.argv) == 3 else sample_configs.keys()

        user_repo = 'DingLi23'
        branch = 'pipelining'
        # user_repo = 'youyinnn'

        # generate the plotting.ipynb if not exist
        # if True:
        for sample_name in sample_list:
            p_nb_file = path.join(data_dir, exp_name, f'{exp_name}_{sample_name}_plotting.ipynb')
            if force or (not path.exists(p_nb_file)):
                print(f'{"Force to g" if force else "G"}enerating plotting notebook for {exp_name} at {p_nb_file}')

                nb = nbf.v4.new_notebook()
                nb.metadata.kernelspec = {
                    "display_name": "Python 3",
                    "name": "python3"
                }
                nb.metadata.language_info = {
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    },
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.10.0"
                }
                open_in_colab_href = f'<a href="https://colab.research.google.com/github/{user_repo}/s2search/blob/{branch}/pipelining/{exp_name}/{exp_name}_{sample_name}_plotting.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'
                exp_des = f'### Experiment Description\n\n{description.strip()}\n\n> This notebook is for experiment \\<{exp_name}\\> and data sample \\<{sample_name}\\>.'

                init_md = '### Initialization'
                init_code = f'''%load_ext autoreload
%autoreload 2
import numpy as np, sys, os
in_colab = 'google.colab' in sys.modules
# fetching code and data(if you are using colab
if in_colab:
    !rm -rf s2search
    !git clone --branch pipelining https://github.com/youyinnn/s2search.git
    sys.path.insert(1, './s2search')
    %cd s2search/pipelining/{exp_name}/
'''

                loading_data_md = '### Loading data'
                loading_data_code = f'''sys.path.insert(1, '../../')
import numpy as np, sys, os
from getting_data import get
from feature_masking import masking_options

sample_data_and_config_arr = get('{exp_name}', '{sample_name}')

for sample_data_and_config in sample_data_and_config_arr:
    y_values = []
    sample_origin_npy = sample_data_and_config['origin']
    for array in sample_data_and_config['feature_stack']:

        # define your y axis value here
        y_value = np.absolute((sample_origin_npy - array) / sample_origin_npy)
        # y_value = sample_origin_npy - array
        y_values.append(y_value)

    y_values = np.array(y_values)
    sample_data_and_config['y_values'] = y_values
'''

                plot_data_md = "### Plot the data"
                plot_data_code = '''import matplotlib.pyplot as plt
def plot_scores_d(sample_name, y_values, sample_origin_npy, query, sample_masking_option_keys): 
  plt.figure(figsize=(20, 15), dpi=80)
  i = 0
  for key in sample_masking_option_keys:
    plt.scatter(
        sample_origin_npy,
        y_values[i],
        c=masking_options[key]['color'], 
        marker=masking_options[key]['marker'],
        label=masking_options[key]['plot_legend']
    )
    i += 1

  plt.xlabel('Orginal Ranking Score',fontsize=16)
  plt.ylabel('$y = \\\\frac{|Score_0 - Score\_feature_j|}{Score_0}$', fontsize=16)
  plt.title(f'Distrubution of Ranking Score and Masking Features Difference\\nfor \\'{sample_name}\\', with query \\'{query}\\'', fontsize=20, pad=20)
  x_max = 10
  x_min = 0
  x_pace = 1
  y_max = 10
  y_min = 0
  y_pace = 1
  plt.xticks(np.arange(x_min, x_max, x_pace), size = 9) 
  plt.yticks(np.arange(y_min, y_max, y_pace), size = 9)
  plt.ylim(y_min, y_max)
  plt.xlim(x_min, x_max)
  plt.legend(prop={'size': 10})
  plt.savefig(os.path.join('.', f'{sample_name}.png'), facecolor='white', transparent=False)
  plt.show()

for sample_data_and_config in sample_data_and_config_arr:
  sample_and_task_name = sample_data_and_config['sample_and_task_name']
  sample_origin_npy = sample_data_and_config['origin']
  y_values = sample_data_and_config['y_values']

  valid_data_count = 0
  total_data_count = 0
  for d in sample_origin_npy:
    total_data_count += 1
    if d > -10:
      valid_data_count += 1
  print(f'({valid_data_count}/{total_data_count}) ({round((valid_data_count / total_data_count) * 100, 2)}%) valid data for {sample_and_task_name} (original score is greater than -10)')

  sample_query = sample_data_and_config['query']
  sample_masking_option_keys = sample_data_and_config['masking_option_keys']
  plot_scores_d(sample_and_task_name, y_values, sample_origin_npy, sample_query, sample_masking_option_keys)
'''

                nb['cells'] = [
                            nbf.v4.new_markdown_cell(open_in_colab_href),
                            nbf.v4.new_markdown_cell(exp_des),
                            
                            nbf.v4.new_markdown_cell(init_md),
                            nbf.v4.new_code_cell(init_code),                    
                            
                            nbf.v4.new_markdown_cell(loading_data_md),
                            nbf.v4.new_code_cell(loading_data_code),

                            nbf.v4.new_markdown_cell(plot_data_md),
                            nbf.v4.new_code_cell(plot_data_code),
                        ]
                nbf.write(nb, str(p_nb_file))
