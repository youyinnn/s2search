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

ale_xy = {{}}

for f in f_list:
    file = os.path.join('.', 'scores', f'{{sample_name}}_pdp_{{f}}.npz')
    if os.path.exists(file):
        feature_pdp_data = np.load(file)['arr_0']
        
        ale_xy[f] = {{
            'y': feature_pdp_data,
            'numerical': True
        }}
        if f == 'year':
            ale_xy[f]['x'] = list(range(1900, 2030))
        elif f == 'n_citations':
            ale_xy[f]['x'] = list(range(0, 11000, 100))
        else:
            ale_xy[f]['y'] = np.sort(feature_pdp_data)
            ale_xy[f]['x'] = list(range(len(feature_pdp_data)))
            ale_xy[f]['numerical'] = False
            
        ale_xy[f]['weird'] = feature_pdp_data[len(feature_pdp_data) - 1] > 30
'''

            plot_data_md = "### PDP"
            plot_data_code = '''import matplotlib.pyplot as plt

categorical_plot_conf = [
    {
        'xlabel': 'Title',
        'ylabel': 'Scores',
        'ale_xy': ale_xy['title']
    },
    {
        'xlabel': 'Abstract',
        'ale_xy': ale_xy['abstract']
    },    
    {
        'xlabel': 'Authors',
        'ale_xy': ale_xy['authors']
    },
    {
        'xlabel': 'Venue',
        'ale_xy': ale_xy['venue'],
        # 'zoom': {
        #     'inset_axes': [0.15, 0.45, 0.47, 0.47],
        #     'x_limit': [950, 1010],
        #     'y_limit': [-9, 7],
        #     'connects': [True, True, False, False]
        # }
    },
]

numerical_plot_conf = [
    {
        'xlabel': 'Year',
        'ylabel': 'Scores',
        'ale_xy': ale_xy['year']
    },
    {
        'xlabel': 'Citation Count',
        'ale_xy': ale_xy['n_citations'],
        # 'zoom': {
        #     'inset_axes': [0.5, 0.2, 0.47, 0.47],
        #     'x_limit': [-100, 1000],
        #     'y_limit': [-7.3, -6.2],
        #     'connects': [False, False, True, True]
        # }
    }
]

def pdp_plot(confs, title):
    fig, axes = plt.subplots(nrows=1, ncols=len(confs), figsize=(20, 5), dpi=100)
    subplot_idx = 0
    plt.suptitle(title, fontsize=20, fontweight='bold')
    # plt.autoscale(False)
    for conf in confs:
        axess = axes if len(confs) == 1 else axes[subplot_idx]

        axess.plot(conf['ale_xy']['x'], conf['ale_xy']['y'])
        axess.grid(alpha = 0.4)

        if ('ylabel' in conf):
            axess.set_ylabel(conf.get('ylabel'), fontsize=20, labelpad=10)
        
        axess.set_xlabel(conf['xlabel'], fontsize=16, labelpad=10)
        
        if not (conf['ale_xy']['weird']):
            if (conf['ale_xy']['numerical']):
                # axess.set_ylim([-1, 3])
                pass
            else:
                axess.set_ylim([-15, 10])
                pass
                
        if 'zoom' in conf:
            axins = axess.inset_axes(conf['zoom']['inset_axes']) 
            axins.plot(conf['ale_xy']['x'], conf['ale_xy']['y'])
            axins.set_xlim(conf['zoom']['x_limit'])
            axins.set_ylim(conf['zoom']['y_limit'])
            axins.grid(alpha=0.3)
            rectpatch, connects = axess.indicate_inset_zoom(axins)
            connects[0].set_visible(conf['zoom']['connects'][0])
            connects[1].set_visible(conf['zoom']['connects'][1])
            connects[2].set_visible(conf['zoom']['connects'][2])
            connects[3].set_visible(conf['zoom']['connects'][3])
            
        subplot_idx += 1

pdp_plot(categorical_plot_conf, "PDPs for four categorical features")
plt.savefig(os.path.join('.', 'plot', f'{sample_name}-categorical.png'), facecolor='white', transparent=False, bbox_inches='tight')

# second fig
pdp_plot(numerical_plot_conf, "PDPs for two numerical features")
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
