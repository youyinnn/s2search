import os
import os.path as path
import yaml
import sys
import nbformat as nbf

data_dir = './pipelining'
user_repo = 'DingLi23'
branch = 'pipelining'
# user_repo = 'youyinnn'


def gen_for_normal(exp_name, description, sample_list):
    for sample_name in sample_list:
        p_nb_file = path.join(data_dir, exp_name,
                              f'{exp_name}_{sample_name}_plotting.ipynb')
        if (not path.exists(p_nb_file)):
            print(
                f'Generating plotting notebook for {exp_name} at {p_nb_file}')

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

pic_dir = os.path.join('.', 'plot')
if not os.path.exists(pic_dir):
    os.mkdir(pic_dir)
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
def plot_scores_d(sample_name, y_values, sample_origin_npy, query, sample_masking_option_keys, sample_task_number): 
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
#   plt.xticks(np.arange(x_min, x_max, x_pace), size = 9) 
#   plt.yticks(np.arange(y_min, y_max, y_pace), size = 9)
#   plt.ylim(y_min, y_max)
#   plt.xlim(x_min, x_max)
  plt.legend(prop={'size': 16})
  plt.savefig(os.path.join('.', 'plot', f'{sample_name}-t{sample_task_number}.png'), facecolor='white', transparent=False)
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
  sample_task_number = sample_data_and_config['task_number']
  plot_scores_d(sample_and_task_name, y_values, sample_origin_npy, sample_query, sample_masking_option_keys, sample_task_number)

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


def gen_for_pdp(exp_name, description, sample_list):
    for sample_name in sample_list:
        p_nb_file = path.join(data_dir, exp_name,
                              f'{exp_name}_{sample_name}_plotting.ipynb')
        if (not path.exists(p_nb_file)):
            print(
                f'Generating plotting notebook for {exp_name} at {p_nb_file}')

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

pic_dir = os.path.join('.', 'plot')
if not os.path.exists(pic_dir):
    os.mkdir(pic_dir)
'''

            loading_data_md = '### Loading data'
            loading_data_code = f'''sys.path.insert(1, '../../')
import numpy as np, sys, os

sample_name = '{sample_name}'

f_list = ['title', 'abstract', 'venue', 'authors', 'year', 'n_citations']

pdp_data_map = {{}}
weird_score = {{}}

for f in f_list:
    feature_pdp_data = np.load(os.path.join('.', 'scores', f'{{sample_name}}_pdp_{{f}}.npz'))['arr_0']
    # print(len(feature_pdp_data))
    pdp_data_map[f] = feature_pdp_data
    for score in feature_pdp_data:
        if score > 20:
            weird_score[f] = ''
'''

            plot_data_md = "### PDP"
            plot_data_code = '''import matplotlib.pyplot as plt

categorical_plot_conf = {
    'title': {
        'xlabel': 'Title',
        'ylabel': 'Scores',
        'x': list(range(len(pdp_data_map['title']))),
        'y': np.sort(pdp_data_map['title']),
        'yticks': list(range(-14, 10, 1)),
    },
    'abstract': {
        'xlabel': 'Abstract',
        'x': list(range(len(pdp_data_map['abstract']))),
        'y': np.sort(pdp_data_map['abstract']),
        'yticks': list(range(-14, 10, 1)),
    },    
    'authors': {
        'xlabel': 'Authors',
        'x': list(range(len(pdp_data_map['authors']))),
        'y': np.sort(pdp_data_map['authors']),
        'yticks': list(range(-14, 10, 1)),
    },
    'venue': {
        'xlabel': 'Venue',
        'x': list(range(len(pdp_data_map['venue']))),
        'y': np.sort(pdp_data_map['venue']),
        'yticks': list(range(-14, 10, 1)),
    },
}

numerical_plot_conf = {
    'year': {
        'xlabel': 'Year',
        'ylabel': 'Scores',
        'x': list(range(1800, 2050)),
        'y': pdp_data_map['year'],
        'xticks': list(range(1800, 2050, 50)),
        'yticks': list(range(-14, 10, 1)),
    },
    'n_citations': {
        'xlabel': 'Citation Count',
        'x': list(range(0, 11000, 50)),
        'y': pdp_data_map['n_citations'],
        'xticks': list(range(0, 11000, 2000)),
        'yticks': list(range(-14, 10, 1)),
    }
}

def pdp_plot(confs):
    idx = 1
    for key in confs.keys():
        conf = confs[key]
        plt.subplot(1, len(confs), idx)
        plt.plot(conf['x'], conf['y'])

        plt.xlabel(conf['xlabel'], fontsize=20, labelpad=20)
        if (conf.get('ylabel') != None):
            plt.ylabel(conf.get('ylabel'), fontsize=20, labelpad=20)

        if conf.get('xticks') != None:
            plt.xticks(conf.get('xticks'), fontsize=14)
        else:
            plt.xticks([], fontsize=14)
            
        if weird_score.get(key) == None:
            plt.yticks(list(range(-14, 10, 1)), fontsize=14)
        else:
            plt.yticks(fontsize=14)
            
        plt.grid()
        idx += 1

plt.figure(figsize=(20, 10), dpi=100)
plt.suptitle("PDPs for four categorical features", y=0.94, fontsize=20, fontweight='bold')
pdp_plot(categorical_plot_conf)
plt.savefig(os.path.join('.', 'plot', f'{sample_name}-categorical.png'), facecolor='white', transparent=False, bbox_inches='tight')

# second fig
plt.figure(figsize=(20, 10), dpi=100)
plt.suptitle("PDPs for two numerical features", y=0.94, fontsize=20, fontweight='bold')
pdp_plot(numerical_plot_conf)
plt.savefig(os.path.join('.', 'plot', f'{sample_name}-numerical.png'), facecolor='white', transparent=False, bbox_inches='tight')

plt.show()
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


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        exp_name = sys.argv[1]
        exp_dir = path.join(data_dir, exp_name)
        if not os.path.exists(exp_dir):
            print(f'No {exp_name} experiment')
            exit()

        conf_path = path.join(exp_dir, 'conf.yml')
        with open(conf_path, 'r') as f:
            conf = yaml.safe_load(f)
            description = conf.get('description')
            sample_configs = conf.get('samples')
            sample_list = sample_configs.keys()

        # generate the plotting.ipynb if not exist
        if exp_name.startswith('exp'):
            gen_for_normal(exp_name, description, sample_list)

        if exp_name.startswith('pdp'):
            gen_for_pdp(exp_name, description, sample_list)
